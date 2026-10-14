"""Employee Attendance & Analytics API."""
import os
import re
import calendar
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Optional, Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Path
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pymongo import MongoClient, ASCENDING, DESCENDING, ReturnDocument
from pymongo.errors import DuplicateKeyError, PyMongoError
from bson import ObjectId

load_dotenv()
client = MongoClient(os.getenv("MONGO_URI", "mongodb://localhost:27017"), serverSelectionTimeoutMS=3000)
db = client[os.getenv("MONGO_DB", "attendance_db")]
IST = timezone(timedelta(hours=5, minutes=30))
UTC = timezone.utc
MIN_MS = 100_000_000_000
MAX_MS = 4_102_444_800_000
PRESENCE = ["PRESENT", "WFH", "ON_DUTY"]
STATUSES = ["PRESENT", "ABSENT", "LEAVE", "WFH", "ON_DUTY"]
app = FastAPI(title="Employee Attendance & Analytics API", version="2.0.0")


@app.on_event("startup")
def create_indexes():
    db.employees.create_index([("emp_code", ASCENDING)], unique=True, name="uq_employee_code")
    db.employees.create_index([("department", ASCENDING), ("joined_on", ASCENDING)], name="department_joined_on")
    db.employees.create_index([("joined_on", ASCENDING)], name="joined_on_idx")
    db.attendance_logs.create_index([("emp_code", ASCENDING), ("date", ASCENDING)], unique=True, name="uq_attendance_employee_date")
    db.attendance_logs.create_index([("date", DESCENDING), ("emp_code", ASCENDING)], name="attendance_date_employee")
    db.attendance_logs.create_index([("emp_code", ASCENDING), ("date", ASCENDING), ("status", ASCENDING)], name="attendance_employee_date_status")
    db.attendance_logs.create_index([("date", ASCENDING), ("late_minutes", DESCENDING)], name="attendance_date_late")


def _millis(dt: Optional[datetime]):
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)  # PyMongo returns naive UTC datetimes by default.
    return int(dt.timestamp() * 1000)


def _from_ms(value: int) -> datetime:
    return datetime.fromtimestamp(value / 1000, tz=UTC).replace(microsecond=0)


def _truncate(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC).replace(microsecond=0)


def _api_record(doc: dict) -> dict:
    out = dict(doc)
    out.pop("_id", None)
    for field in ("punch_in", "punch_out"):
        out[field] = _millis(out.get(field))
    out["work_hours"] = out.get("work_hours")
    out["late_minutes"] = int(out.get("late_minutes") or 0)
    out["overtime_minutes"] = int(out.get("overtime_minutes") or 0)
    out["half_day"] = bool(out.get("half_day", False))
    history = []
    for entry in out.get("history") or []:
        e = dict(entry)
        e["at"] = _millis(e.get("at"))
        changes = {}
        for key, change in (e.get("changes") or {}).items():
            c = dict(change)
            if key in ("punch_in", "punch_out"):
                c["from"] = _millis(c.get("from"))
                c["to"] = _millis(c.get("to"))
            changes[key] = c
        e["changes"] = changes
        history.append(e)
    out["history"] = history
    return out


def _employee_api(doc: dict) -> dict:
    out = dict(doc)
    out.pop("_id", None)
    out["created_at"] = _millis(out.get("created_at"))
    return out


def _date(value: str, field: str = "date") -> date:
    try:
        if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
            raise ValueError
        return date.fromisoformat(value)
    except (ValueError, TypeError):
        raise HTTPException(422, f"{field} must be a valid YYYY-MM-DD date")


def _month(value: str) -> tuple[str, str, date, date]:
    if not re.fullmatch(r"\d{4}-(0[1-9]|1[0-2])", value or ""):
        raise HTTPException(422, "month must use YYYY-MM format")
    y, m = map(int, value.split("-"))
    first = date(y, m, 1)
    last = date(y, m, calendar.monthrange(y, m)[1])
    return value, first.isoformat(), last.isoformat(), last


def _validate_ms(value):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < MIN_MS or value > MAX_MS:
        raise ValueError(f"timestamp must be an integer epoch-millisecond value from {MIN_MS} to {MAX_MS}")
    return value


def _shift_parts(value: str) -> tuple[int, int]:
    if not isinstance(value, str) or not re.fullmatch(r"(?:[01]\d|2[0-3]):[0-5]\d", value):
        raise HTTPException(422, "shift times must use HH:MM")
    return tuple(map(int, value.split(":")))


def _attendance_day(ts: datetime, emp: dict) -> date:
    local = ts.astimezone(IST)
    start_h, start_m = _shift_parts(emp["shift_start"])
    end_h, end_m = _shift_parts(emp["shift_end"])
    start_minutes, end_minutes = start_h * 60 + start_m, end_h * 60 + end_m
    d = local.date()
    if end_minutes <= start_minutes and local.hour * 60 + local.minute < end_minutes:
        d -= timedelta(days=1)
    return d


def compute_late_minutes(punch_in: datetime, shift_start: str, attendance_date: str) -> int:
    h, m = _shift_parts(shift_start)
    d = _date(attendance_date)
    local = punch_in.astimezone(IST)
    shift_start_dt = datetime.combine(d, time(h, m), IST)
    # For overnight shifts, the shift starts on the attendance date even if punch-in is after midnight.
    elapsed_seconds = (local - shift_start_dt).total_seconds()
    if elapsed_seconds > 600:
        return int(elapsed_seconds // 60)
    return 0


def _derived(punch_in: Optional[datetime], punch_out: Optional[datetime], emp: dict, attendance_date: str, status: str) -> dict:
    if status not in PRESENCE or punch_in is None:
        return {"late_minutes": 0, "work_hours": None, "overtime_minutes": 0, "half_day": False}
    pin = _truncate(punch_in)
    late = compute_late_minutes(pin, emp["shift_start"], attendance_date)
    if punch_out is None:
        return {"late_minutes": late, "work_hours": None, "overtime_minutes": 0, "half_day": False}
    pout = _truncate(punch_out)
    seconds = int((pout - pin).total_seconds())
    hours = float((Decimal(seconds) / Decimal(3600)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))
    eh, em = _shift_parts(emp["shift_end"])
    sh, sm = _shift_parts(emp["shift_start"])
    shift_end_date = _date(attendance_date)
    if (eh * 60 + em) <= (sh * 60 + sm):
        shift_end_date += timedelta(days=1)
    end_dt = datetime.combine(shift_end_date, time(eh, em), IST)
    overtime_raw = int((pout.astimezone(IST) - end_dt).total_seconds() // 60)
    overtime = max(0, overtime_raw) if overtime_raw >= 30 else 0
    return {"late_minutes": late, "work_hours": hours, "overtime_minutes": overtime, "half_day": hours < 4.50}


def _validate_page(page: int, page_size: int):
    if page < 1 or page_size < 1 or page_size > 100:
        raise HTTPException(422, "page must be >= 1 and page_size must be between 1 and 100")


class EmployeeIn(BaseModel):
    model_config = ConfigDict(extra="ignore")
    emp_code: str = Field(pattern=r"^EMP\d{4,6}$")
    name: str = Field(min_length=1, max_length=100)
    email: str = Field(max_length=120, pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
    department: str = Field(min_length=1, max_length=50)
    shift_start: str = Field(default="09:30", pattern=r"^(?:[01]\d|2[0-3]):[0-5]\d$")
    shift_end: str = Field(default="18:30", pattern=r"^(?:[01]\d|2[0-3]):[0-5]\d$")
    joined_on: str

    @field_validator("joined_on")
    @classmethod
    def valid_joined_on(cls, value):
        date.fromisoformat(value)
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
            raise ValueError("joined_on must use YYYY-MM-DD")
        return value

    @model_validator(mode="after")
    def shifts_differ(self):
        if self.shift_start == self.shift_end:
            raise ValueError("shift_start must differ from shift_end")
        return self


class PunchInIn(BaseModel):
    model_config = ConfigDict(extra="ignore")
    emp_code: str
    punched_at: Optional[int] = None
    status: Literal["PRESENT", "WFH", "ON_DUTY"] = "PRESENT"

    @field_validator("punched_at", mode="before")
    @classmethod
    def valid_punched_at(cls, value):
        return _validate_ms(value)


class PunchOutIn(BaseModel):
    model_config = ConfigDict(extra="ignore")
    emp_code: str
    punched_at: Optional[int] = None

    @field_validator("punched_at", mode="before")
    @classmethod
    def valid_punched_at(cls, value):
        return _validate_ms(value)


class RegularizeIn(BaseModel):
    model_config = ConfigDict(extra="ignore")
    status: Optional[Literal["PRESENT", "ABSENT", "LEAVE", "WFH", "ON_DUTY"]] = None
    punch_in: Optional[int] = None
    punch_out: Optional[int] = None
    reason: str = Field(min_length=5, max_length=200)
    regularized_by: str = Field(min_length=1, max_length=50)

    @field_validator("punch_in", "punch_out", mode="before")
    @classmethod
    def valid_times(cls, value):
        return _validate_ms(value)


@app.get("/health")
def health():
    try:
        client.admin.command("ping")
        return {"status": "ok"}
    except PyMongoError:
        raise HTTPException(503, "MongoDB is unavailable")


@app.post("/employees", status_code=201)
def create_employee(body: EmployeeIn):
    doc = body.model_dump()
    doc["created_at"] = datetime.now(UTC).replace(microsecond=0)
    try:
        db.employees.insert_one(doc)
    except DuplicateKeyError:
        raise HTTPException(409, "emp_code already exists")
    return _employee_api(doc)


@app.get("/employees")
def list_employees(department: Optional[str] = None, page: int = Query(1, ge=1), page_size: int = Query(20, ge=1, le=100)):
    q = {"department": department} if department is not None else {}
    skip = (page - 1) * page_size
    total = db.employees.count_documents(q)
    items = [_employee_api(x) for x in db.employees.find(q, {"_id": 0}).sort("emp_code", ASCENDING).skip(skip).limit(page_size)]
    return {"items": items, "total": total, "page": page, "page_size": page_size}


@app.post("/attendance/punch-in", status_code=201)
def punch_in(body: PunchInIn):
    emp = db.employees.find_one({"emp_code": body.emp_code}, {"_id": 0})
    if not emp:
        raise HTTPException(404, "employee not found")
    ts = _truncate(_from_ms(body.punched_at) if body.punched_at is not None else datetime.now(UTC))
    attendance_date = _attendance_day(ts, emp).isoformat()
    derived = _derived(ts, None, emp, attendance_date, body.status)
    doc = {"emp_code": body.emp_code, "date": attendance_date, "status": body.status,
           "punch_in": ts, "punch_out": None, **derived, "history": []}
    try:
        db.attendance_logs.insert_one(doc)
    except DuplicateKeyError:
        raise HTTPException(409, "already punched in for this date")
    return _api_record(doc)


@app.post("/attendance/punch-out")
def punch_out(body: PunchOutIn):
    emp = db.employees.find_one({"emp_code": body.emp_code}, {"_id": 0})
    if not emp:
        raise HTTPException(404, "employee not found")
    ts = _truncate(_from_ms(body.punched_at) if body.punched_at is not None else datetime.now(UTC))
    ts_local = ts.astimezone(IST)
    lower = ts - timedelta(hours=24)
    candidates = db.attendance_logs.find({"emp_code": body.emp_code, "punch_in": {"$ne": None, "$lte": ts}}).sort("punch_in", DESCENDING).limit(1)
    rec = next(candidates, None)
    if not rec:
        raise HTTPException(404, "no punch-in found")
    if rec.get("punch_out") is not None:
        raise HTTPException(409, "record already punched out")
    # PyMongo returns BSON datetimes as naive UTC by default. Normalize the
    # stored punch-in to timezone-aware UTC before comparing it with `ts`.
    punch_in = rec["punch_in"]
    if punch_in.tzinfo is None:
        punch_in = punch_in.replace(tzinfo=UTC)
    if ts <= punch_in or ts > punch_in + timedelta(hours=24):
        raise HTTPException(422, "punched_at must be after punch-in and within 24 hours")
    derived = _derived(punch_in, ts, emp, rec["date"], rec["status"])
    result = db.attendance_logs.find_one_and_update(
        {"_id": rec["_id"], "punch_out": None, "punch_in": rec["punch_in"]},
        {"$set": {"punch_out": ts, **derived}}, return_document=ReturnDocument.AFTER)
    if result is None:
        raise HTTPException(409, "record already punched out")
    return _api_record(result)


@app.get("/attendance")
def list_attendance(emp_code: Optional[str] = None, date_from: Optional[str] = None, date_to: Optional[str] = None,
                    status: Optional[Literal["PRESENT", "ABSENT", "LEAVE", "WFH", "ON_DUTY"]] = None, page: int = Query(1, ge=1), page_size: int = Query(20, ge=1, le=100)):
    q = {}
    if emp_code: q["emp_code"] = emp_code
    if status: q["status"] = status
    if date_from:
        _date(date_from, "date_from")
        q.setdefault("date", {})["$gte"] = date_from
    if date_to:
        _date(date_to, "date_to")
        q.setdefault("date", {})["$lte"] = date_to
    if date_from and date_to and date_from > date_to:
        raise HTTPException(422, "date_from must be on or before date_to")
    skip = (page - 1) * page_size
    total = db.attendance_logs.count_documents(q)
    docs = db.attendance_logs.find(q).sort([("date", DESCENDING), ("emp_code", ASCENDING)]).skip(skip).limit(page_size)
    return {"items": [_api_record(d) for d in docs], "total": total, "page": page, "page_size": page_size}


@app.patch("/attendance/{emp_code}/{date}")
def regularize_attendance(emp_code: str, date: str, body: RegularizeIn):
    _date(date)
    emp = db.employees.find_one({"emp_code": emp_code}, {"_id": 0})
    if not emp:
        raise HTTPException(404, "employee not found")
    rec = db.attendance_logs.find_one({"emp_code": emp_code, "date": date})
    if not rec:
        raise HTTPException(404, "attendance record not found")
    old_status = rec["status"]
    new_status = body.status if body.status is not None else old_status
    old_in, old_out = rec.get("punch_in"), rec.get("punch_out")
    if new_status in ("ABSENT", "LEAVE"):
        if body.punch_in is not None or body.punch_out is not None:
            raise HTTPException(422, "ABSENT and LEAVE cannot have punch times")
        new_in = new_out = None
    else:
        new_in = _truncate(_from_ms(body.punch_in)) if body.punch_in is not None else old_in
        new_out = _truncate(_from_ms(body.punch_out)) if body.punch_out is not None else old_out
        if new_in is None:
            raise HTTPException(422, "presence status requires punch_in")
        if _attendance_day(new_in, emp).isoformat() != date:
            raise HTTPException(422, "punch_in must belong to the record attendance date")
        if new_out is not None and (new_out <= new_in or new_out > new_in + timedelta(hours=24)):
            raise HTTPException(422, "punch_out must be after punch_in and within 24 hours")
    derived = _derived(new_in, new_out, emp, date, new_status)
    before = {"status": old_status, "punch_in": old_in, "punch_out": old_out,
              "work_hours": rec.get("work_hours"), "late_minutes": rec.get("late_minutes", 0),
              "overtime_minutes": rec.get("overtime_minutes", 0), "half_day": rec.get("half_day", False)}
    after = {"status": new_status, "punch_in": new_in, "punch_out": new_out, **derived}
    changes = {}
    for key in before:
        if before[key] != after[key]:
            fr, to = before[key], after[key]
            changes[key] = {"from": fr, "to": to}
    if not changes:
        raise HTTPException(422, "request does not change the record")
    entry = {"at": datetime.now(UTC).replace(microsecond=0), "by": body.regularized_by, "reason": body.reason, "changes": changes}
    # Optimistic concurrency: only update if the values read above are still current.
    query = {"_id": rec["_id"], "status": old_status, "punch_in": old_in, "punch_out": old_out}
    query["history"] = rec.get("history", [])
    result = db.attendance_logs.find_one_and_update(query, {"$set": {"status": new_status, "punch_in": new_in, "punch_out": new_out, **derived}, "$push": {"history": entry}}, return_document=ReturnDocument.AFTER)
    if result is None:
        raise HTTPException(409, "record changed concurrently; retry")
    return _api_record(result)


def _working_days(joined_on: str, month_start: str, month_end: str) -> int:
    start = max(_date(joined_on, "joined_on"), _date(month_start), date.fromisoformat(month_start))
    end = _date(month_end)
    if start > end: return 0
    count = 0
    current = start
    while current <= end:
        if current.weekday() < 5: count += 1
        current += timedelta(days=1)
    return count


def _monthly_pipeline(emp_code: str, start: str, end_exclusive: str):
    return [
        {"$match": {"emp_code": emp_code, "date": {"$gte": start, "$lt": end_exclusive}}},
        {"$set": {"_day": {"$dateFromString": {"dateString": "$date"}}, "_late": {"$ifNull": ["$late_minutes", 0]}, "_ot": {"$ifNull": ["$overtime_minutes", 0]}, "_half": {"$ifNull": ["$half_day", False]}}},
        {"$group": {"_id": None,
            "present_days": {"$sum": {"$cond": [{"$and": [{"$in": ["$status", PRESENCE]}, {"$lt": [{"$dayOfWeek": "$_day"}, 7]}, {"$gt": [{"$dayOfWeek": "$_day"}, 1]}]}, {"$cond": ["$_half", 0.5, 1]}, 0]}},
            "leave_days": {"$sum": {"$cond": [{"$eq": ["$status", "LEAVE"]}, 1, 0]}},
            "late_count": {"$sum": {"$cond": [{"$gt": ["$_late", 0]}, 1, 0]}},
            "total_late_minutes": {"$sum": "$_late"}, "total_overtime_minutes": {"$sum": "$_ot"}}}
    ]


@app.get("/analytics/employees/{emp_code}/monthly")
def employee_monthly(emp_code: str, month: str = Query(..., pattern=r"^\d{4}-(0[1-9]|1[0-2])$")):
    month, start, month_end, last_day = _month(month)
    emp = db.employees.find_one({"emp_code": emp_code}, {"_id": 0})
    if not emp: raise HTTPException(404, "employee not found")
    end_exclusive = (last_day + timedelta(days=1)).isoformat()
    rows = list(db.attendance_logs.aggregate(_monthly_pipeline(emp_code, start, end_exclusive)))
    stats = rows[0] if rows else {}
    working = _working_days(emp["joined_on"], start, month_end)
    present = float(stats.get("present_days", 0))
    pct = None if working == 0 else float((Decimal(str(present / working * 100))).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP))
    return {"emp_code": emp_code, "month": month, "working_days": working, "present_days": present,
            "leave_days": int(stats.get("leave_days", 0)), "late_count": int(stats.get("late_count", 0)),
            "total_late_minutes": int(stats.get("total_late_minutes", 0)), "total_overtime_minutes": int(stats.get("total_overtime_minutes", 0)), "attendance_pct": pct}


def _department_summary_pipeline(start: str, end_exclusive: str, department: Optional[str]):
    match = {"joined_on": {"$lt": end_exclusive}}
    if department is not None: match["department"] = department
    return [
        {"$match": match},
        {"$lookup": {"from": "attendance_logs", "let": {"code": "$emp_code"}, "pipeline": [{"$match": {"$expr": {"$and": [{"$eq": ["$emp_code", "$$code"]}, {"$gte": ["$date", start]}, {"$lt": ["$date", end_exclusive]}]}}}], "as": "logs"}},
        {"$unwind": {"path": "$logs", "preserveNullAndEmptyArrays": True}},
        {"$set": {"_log": {"$ifNull": ["$logs", {}]}, "_day": {"$cond": [{"$ne": ["$logs", None]}, {"$dateFromString": {"dateString": "$logs.date"}}, None] } }},
        {"$group": {"_id": {"department": "$department", "emp_code": "$emp_code"},
            "headcount": {"$sum": 1},
            "present_days": {"$sum": {"$cond": [{"$and": [{"$in": ["$_log.status", PRESENCE]}, {"$ne": ["$_day", None]}, {"$gt": [{"$dayOfWeek": "$_day"}, 1]}, {"$lt": [{"$dayOfWeek": "$_day"}, 7]}]}, {"$cond": [{"$ifNull": ["$_log.half_day", False]}, 0.5, 1]}, 0]}},
            "work_sum": {"$sum": {"$cond": [{"$and": [{"$in": ["$_log.status", PRESENCE]}, {"$ne": ["$_log.work_hours", None]}]}, "$_log.work_hours", 0]}},
            "work_count": {"$sum": {"$cond": [{"$and": [{"$in": ["$_log.status", PRESENCE]}, {"$ne": ["$_log.work_hours", None]}]}, 1, 0]}},
            "late_count": {"$sum": {"$cond": [{"$gt": [{"$ifNull": ["$_log.late_minutes", 0]}, 0]}, 1, 0]}},
            "total_late_minutes": {"$sum": {"$ifNull": ["$_log.late_minutes", 0]}},
            "leave_count": {"$sum": {"$cond": [{"$eq": ["$_log.status", "LEAVE"]}, 1, 0]}},
            "on_duty_count": {"$sum": {"$cond": [{"$eq": ["$_log.status", "ON_DUTY"]}, 1, 0]}}}},
        {"$group": {"_id": "$_id.department", "headcount": {"$sum": 1}, "present_days": {"$sum": "$present_days"}, "work_sum": {"$sum": "$work_sum"}, "work_count": {"$sum": "$work_count"}, "late_count": {"$sum": "$late_count"}, "total_late_minutes": {"$sum": "$total_late_minutes"}, "leave_count": {"$sum": "$leave_count"}, "on_duty_count": {"$sum": "$on_duty_count"}}},
        {"$project": {"_id": 0, "department": "$_id", "headcount": 1, "present_days": 1,
            "avg_work_hours": {"$cond": [{"$gt": ["$work_count", 0]}, {"$round": [{"$divide": ["$work_sum", "$work_count"]}, 2]}, None]},
            "late_count": 1, "total_late_minutes": 1, "leave_count": 1, "on_duty_count": 1}},
        {"$sort": {"department": 1}}
    ]


@app.get("/analytics/departments/summary")
def department_summary(month: str = Query(..., pattern=r"^\d{4}-(0[1-9]|1[0-2])$"), department: Optional[str] = None):
    month, start, last, last_day = _month(month)
    end_exclusive = (last_day + timedelta(days=1)).isoformat()
    pipeline = _department_summary_pipeline(start, end_exclusive, department)
    items = list(db.employees.aggregate(pipeline))
    # The first grouping counts each eligible employee once, including employees with no logs.
    for item in items:
        if item.get("avg_work_hours") is not None:
            item["avg_work_hours"] = float(Decimal(str(item["avg_work_hours"])).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))
        item["present_days"] = float(item.get("present_days", 0))
    return {"month": month, "items": items}


def _leaderboard_pipeline(start: str, end_exclusive: str, department: Optional[str], limit: int):
    pipeline = [
        {"$match": {"date": {"$gte": start, "$lt": end_exclusive}}},
        {"$group": {"_id": "$emp_code", "total_late_minutes": {"$sum": {"$ifNull": ["$late_minutes", 0]}}, "late_count": {"$sum": {"$cond": [{"$gt": [{"$ifNull": ["$late_minutes", 0]}, 0]}, 1, 0]}}}},
        {"$match": {"total_late_minutes": {"$gt": 0}}},
        {"$lookup": {"from": "employees", "localField": "_id", "foreignField": "emp_code", "as": "employee"}},
        {"$unwind": "$employee"},
    ]
    if department is not None: pipeline.append({"$match": {"employee.department": department}})
    pipeline += [
        {"$setWindowFields": {"sortBy": {"total_late_minutes": -1}, "output": {"rank": {"$rank": {}}}}},
        {"$match": {"rank": {"$lte": limit}}},
        {"$sort": {"total_late_minutes": -1, "_id": 1}},
        {"$project": {"_id": 0, "rank": 1, "emp_code": "$_id", "name": "$employee.name", "department": "$employee.department", "total_late_minutes": 1, "late_count": 1}}
    ]
    return pipeline


@app.get("/analytics/leaderboard/late")
def late_leaderboard(month: str = Query(..., pattern=r"^\d{4}-(0[1-9]|1[0-2])$"), limit: int = Query(10, ge=1, le=50), department: Optional[str] = None):
    month, start, last, last_day = _month(month)
    end_exclusive = (last_day + timedelta(days=1)).isoformat()
    items = list(db.attendance_logs.aggregate(_leaderboard_pipeline(start, end_exclusive, department, limit)))
    return {"month": month, "items": items}


def _trend_pipeline(department: str, start_date: date, end_date: date):
    days = (end_date - start_date).days + 1
    start_dt = datetime.combine(start_date, time.min, UTC)
    return [
        {"$match": {"department": department}},
        {"$limit": 1},
        {"$project": {"_id": 0, "dates": {"$map": {
            "input": {"$range": [0, days]}, "as": "i",
            "in": {"$dateAdd": {"startDate": {"$literal": start_dt}, "unit": "day", "amount": "$$i"}}
        }}}},
        {"$unwind": "$dates"},
        {"$set": {"date": {"$dateToString": {"date": "$dates", "format": "%Y-%m-%d", "timezone": "UTC"}}}},
        {"$lookup": {
            "from": "employees", "let": {"day": "$date"},
            "pipeline": [
                {"$match": {"$expr": {"$and": [
                    {"$eq": ["$department", department]}, {"$lte": ["$joined_on", "$$day"]}
                ]}}},
                {"$count": "n"}
            ], "as": "hc"
        }},
        {"$set": {"headcount": {"$ifNull": [{"$arrayElemAt": ["$hc.n", 0]}, 0]}}},
        {"$lookup": {
            "from": "attendance_logs", "let": {"day": "$date"},
            "pipeline": [
                {"$match": {"$expr": {"$eq": ["$date", "$$day"]}}},
                {"$lookup": {"from": "employees", "localField": "emp_code", "foreignField": "emp_code", "as": "e"}},
                {"$unwind": "$e"},
                {"$match": {"e.department": department}},
                {"$group": {"_id": None,
                    "present_count": {"$sum": {"$cond": [
                        {"$in": ["$status", PRESENCE]},
                        {"$cond": [{"$ifNull": ["$half_day", False]}, 0.5, 1]}, 0
                    ]}},
                    "late_count": {"$sum": {"$cond": [
                        {"$gt": [{"$ifNull": ["$late_minutes", 0]}, 0]}, 1, 0
                    ]}}
                }}
            ], "as": "day_stats"
        }},
        {"$set": {
            "present_count": {"$ifNull": [{"$arrayElemAt": ["$day_stats.present_count", 0]}, 0]},
            "late_count": {"$ifNull": [{"$arrayElemAt": ["$day_stats.late_count", 0]}, 0]},
            "is_working_day": {"$and": [
                {"$gte": [{"$dayOfWeek": "$dates"}, 2]},
                {"$lte": [{"$dayOfWeek": "$dates"}, 6]}
            ]}
        }},
        {"$set": {"attendance_rate": {"$cond": [
            {"$and": ["$is_working_day", {"$gt": ["$headcount", 0]}]},
            {"$round": [{"$divide": ["$present_count", "$headcount"]}, 4]}, None
        ]}}},
        {"$setWindowFields": {"sortBy": {"dates": 1}, "output": {
            "moving_avg_7d": {"$avg": "$attendance_rate", "window": {"documents": [-6, 0]}}
        }}},
        {"$project": {"_id": 0, "date": 1, "is_working_day": 1, "headcount": 1,
            "present_count": 1, "late_count": 1, "attendance_rate": 1,
            "moving_avg_7d": {"$cond": [
                {"$eq": ["$moving_avg_7d", None]}, None, {"$round": ["$moving_avg_7d", 4]}
            ]}
        }},
        {"$sort": {"date": 1}}
    ]


@app.get("/analytics/departments/{department}/trend")
def department_trend(department: str, from_date: Optional[str] = Query(None, alias="from"), to_date: Optional[str] = Query(None, alias="to")):
    if from_date is None or to_date is None:
        raise HTTPException(422, "from and to are required")
    start, end = _date(from_date, "from"), _date(to_date, "to")
    if end < start or (end - start).days + 1 > 92:
        raise HTTPException(422, "to must be on/after from and range cannot exceed 92 days")
    if not db.employees.find_one({"department": department}, {"_id": 1}):
        raise HTTPException(404, "department not found")
    items = list(db.employees.aggregate(_trend_pipeline(department, start, end)))
    for item in items:
        for field in ("attendance_rate", "moving_avg_7d"):
            if item.get(field) is not None:
                item[field] = float(Decimal(str(item[field])).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP))
        item["present_count"] = float(item.get("present_count", 0))
    return {"department": department, "items": items}


def _json_safe(value):
    if isinstance(value, dict): return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list): return [_json_safe(v) for v in value]
    if isinstance(value, ObjectId): return str(value)
    if isinstance(value, datetime): return value.isoformat()
    return value


@app.get("/admin/explain/{endpoint}")
def explain_endpoint(endpoint: str, emp_code: Optional[str] = None, month: Optional[str] = None,
                     department: Optional[str] = None, limit: int = Query(10, ge=1, le=50),
                     date_from: Optional[str] = None, date_to: Optional[str] = None, status: Optional[str] = None,
                     page: int = Query(1, ge=1), page_size: int = Query(20, ge=1, le=100),
                     from_date: Optional[str] = Query(None, alias="from"), to_date: Optional[str] = Query(None, alias="to")):
    if endpoint not in {"attendance_list", "employee_monthly", "department_summary", "late_leaderboard", "department_trend"}:
        raise HTTPException(422, "unsupported endpoint")
    _validate_page(page, page_size)
    if endpoint == "attendance_list":
        q = {}
        if emp_code: q["emp_code"] = emp_code
        if status: q["status"] = status
        if date_from:
            _date(date_from, "date_from"); q.setdefault("date", {})["$gte"] = date_from
        if date_to:
            _date(date_to, "date_to"); q.setdefault("date", {})["$lte"] = date_to
        if date_from and date_to and date_from > date_to: raise HTTPException(422, "date_from must be on or before date_to")
        cmd = {"find": "attendance_logs", "filter": q, "sort": {"date": -1, "emp_code": 1}, "skip": (page - 1) * page_size, "limit": page_size}
        coll = "attendance_logs"
    else:
        if not month and endpoint in {"employee_monthly", "department_summary", "late_leaderboard"}:
            raise HTTPException(422, "month is required")
        if endpoint == "employee_monthly":
            if not emp_code: raise HTTPException(422, "emp_code is required")
            _, start, last, last_day = _month(month)
            cmd = {"aggregate": "attendance_logs", "pipeline": _monthly_pipeline(emp_code, start, (last_day + timedelta(days=1)).isoformat()), "cursor": {}}
            coll = "attendance_logs"
        elif endpoint == "department_summary":
            _, start, last, last_day = _month(month)
            cmd = {"aggregate": "employees", "pipeline": _department_summary_pipeline(start, (last_day + timedelta(days=1)).isoformat(), department), "cursor": {}}
            coll = "employees"
        elif endpoint == "late_leaderboard":
            _, start, last, last_day = _month(month)
            cmd = {"aggregate": "attendance_logs", "pipeline": _leaderboard_pipeline(start, (last_day + timedelta(days=1)).isoformat(), department, limit), "cursor": {}}
            coll = "attendance_logs"
        else:
            if not department or not from_date or not to_date: raise HTTPException(422, "department, from and to are required")
            start_d, end_d = _date(from_date, "from"), _date(to_date, "to")
            if end_d < start_d or (end_d - start_d).days + 1 > 92: raise HTTPException(422, "invalid trend date range")
            cmd = {"aggregate": "employees", "pipeline": _trend_pipeline(department, start_d, end_d), "cursor": {}}
            coll = "employees"
    try:
        explain = db.command({"explain": cmd, "verbosity": "executionStats"})
    except PyMongoError as exc:
        raise HTTPException(422, f"could not explain query: {exc.__class__.__name__}")
    return {"endpoint": endpoint, "collection": coll, "explain": _json_safe(explain)}