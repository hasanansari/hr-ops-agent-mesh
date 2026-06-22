"""OpenAI-style tool schemas for Action Agent's mock HR self-service tools.
Structured JSON I/O, defined as data so they can be validated and reused
the same way regardless of which agent or LLM eventually calls them.
"""

TOOL_SCHEMAS = [
    {
        "name": "check_leave_balance",
        "description": "Check an employee's remaining annual leave balance and how much they've taken this quarter.",
        "input_schema": {
            "type": "object",
            "properties": {"employee_id": {"type": "string", "description": "e.g. EMP-0001"}},
            "required": ["employee_id"],
        },
    },
    {
        "name": "apply_for_leave",
        "description": "Submit a leave application for an employee, given a number of days and a start date.",
        "input_schema": {
            "type": "object",
            "properties": {
                "employee_id": {"type": "string"},
                "days": {"type": "integer", "minimum": 1},
                "start_date": {"type": "string", "description": "ISO 8601 date, e.g. 2026-06-15"},
            },
            "required": ["employee_id", "days", "start_date"],
        },
    },
    {
        "name": "fetch_payslip",
        "description": "Fetch a summary of an employee's most recent payslip.",
        "input_schema": {
            "type": "object",
            "properties": {"employee_id": {"type": "string"}},
            "required": ["employee_id"],
        },
    },
]
