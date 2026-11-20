from __future__ import annotations
from pydantic import BaseModel, Field
from .base import Tool


class SendEmailArgs(BaseModel):
    to: str = Field(..., description="Recipient email address.")
    subject: str = Field(..., description="Email subject line.")
    body: str = Field("", description="Email body.")


class SendEmailTool(Tool):
    name = "send_email"
    description = (
        "Send an email to a recipient. This performs a real, irreversible action, "
        "so it is gated behind human confirmation.")
    Args = SendEmailArgs
    idempotent = False
    has_side_effects = True

    def run(self, to: str, subject: str, body: str = "") -> str:
        return f"Email sent to {to}: '{subject}'"
