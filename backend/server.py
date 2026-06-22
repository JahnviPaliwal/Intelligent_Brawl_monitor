"""
Intelligent Brawl Monitor — Backend Server
==========================================
Serves the frontend and handles real email dispatch via /api/alert.
Run: python server.py
Then open: http://localhost:5000
"""

import smtplib
import json
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse
import threading


# ---------- Config (override via environment variables) ----------
SMTP_HOST     = os.environ.get("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT     = int(os.environ.get("SMTP_PORT", 587))
SENDER_EMAIL  = os.environ.get("SENDER_EMAIL", "")       # set this
SENDER_PASS   = os.environ.get("SENDER_PASS", "")        # Gmail app password
RECIPIENT     = os.environ.get("RECIPIENT_EMAIL", "")    # security team email
SERVE_PORT    = int(os.environ.get("PORT", 5000))


# ---------- Email dispatch ----------
def send_alert_email(alert_data: dict, sender: str, password: str, recipient: str) -> dict:
    """Send fight detection alert email via Gmail SMTP."""
    if not sender or not password or not recipient:
        return {"ok": False, "error": "Email not configured. Set SENDER_EMAIL, SENDER_PASS, RECIPIENT_EMAIL env vars."}

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    threat_pct = round(alert_data.get("threat", 0) * 100)
    elbow_angle = alert_data.get("elbowAngle", "—")
    alert_num = alert_data.get("alertNum", "—")

    subject = f"🚨 Brawl Alert [{timestamp}]"

    html_body = f"""
    <html><body style="font-family: sans-serif; color: #222;">
      <div style="max-width:600px; margin:0 auto; border:1px solid #e0e0e0; border-radius:8px; overflow:hidden;">
        <div style="background:#e84040; padding:16px 20px;">
          <h2 style="color:#fff; margin:0;">🚨 Fight Detected — Brawl Monitor Alert #{alert_num}</h2>
        </div>
        <div style="padding:20px;">
          <p style="font-size:16px; margin-bottom:16px;">
            The Intelligent Brawl Monitor has detected a potential physical altercation.
            Immediate response required.
          </p>
          <table style="width:100%; border-collapse:collapse; font-size:14px;">
            <tr style="background:#f5f5f5;">
              <td style="padding:8px 12px; font-weight:600; width:160px;">Timestamp</td>
              <td style="padding:8px 12px;">{timestamp}</td>
            </tr>
            <tr>
              <td style="padding:8px 12px; font-weight:600;">Threat Score</td>
              <td style="padding:8px 12px; color:#e84040; font-weight:700;">{threat_pct}%</td>
            </tr>
            <tr style="background:#f5f5f5;">
              <td style="padding:8px 12px; font-weight:600;">Max Elbow Angle</td>
              <td style="padding:8px 12px;">{elbow_angle}° (≥140° = strike pose)</td>
            </tr>
            <tr>
              <td style="padding:8px 12px; font-weight:600;">Wrists Elevated</td>
              <td style="padding:8px 12px;">{"Yes" if alert_data.get("wristsUp") else "No"}</td>
            </tr>
            <tr style="background:#f5f5f5;">
              <td style="padding:8px 12px; font-weight:600;">Persons Detected</td>
              <td style="padding:8px 12px;">{alert_data.get("numPersons", "—")}</td>
            </tr>
          </table>
          <p style="margin-top:20px; font-size:13px; color:#666;">
            This alert was dispatched automatically by the Intelligent Brawl Monitor system.
            Please investigate the location immediately.
          </p>
        </div>
        <div style="background:#f5f5f5; padding:12px 20px; font-size:12px; color:#888;">
          Intelligent Brawl Monitor · Powered by MediaPipe Pose · Alert #{alert_num}
        </div>
      </div>
    </body></html>
    """

    text_body = (
        f"BRAWL ALERT #{alert_num}\n"
        f"Timestamp: {timestamp}\n"
        f"Threat Score: {threat_pct}%\n"
        f"Elbow Angle: {elbow_angle}°\n"
        f"Wrists Elevated: {'Yes' if alert_data.get('wristsUp') else 'No'}\n"
        f"Persons Detected: {alert_data.get('numPersons', '—')}\n"
    )

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = sender
    msg["To"]      = recipient
    msg.attach(MIMEText(text_body, "plain"))
    msg.attach(MIMEText(html_body, "html"))

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(sender, password)
            server.sendmail(sender, recipient, msg.as_string())
        print(f"[{timestamp}] ✓ Alert #{alert_num} email sent to {recipient}")
        return {"ok": True, "timestamp": timestamp}
    except smtplib.SMTPAuthenticationError:
        return {"ok": False, "error": "SMTP auth failed. Check SENDER_EMAIL and SENDER_PASS (use Gmail App Password)."}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


# ---------- HTTP handler ----------
class Handler(SimpleHTTPRequestHandler):
    """Serves static files + handles /api/alert POST."""

    def __init__(self, *args, **kwargs):
        # Serve from project root (one level up from backend/)
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        super().__init__(*args, directory=root, **kwargs)

    def do_OPTIONS(self):
        self.send_response(200)
        self._cors()
        self.end_headers()

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/alert":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            try:
                data = json.loads(body)
            except Exception:
                data = {}

            # Use values from request body if provided, else fall back to env config
            sender    = data.get("senderEmail")    or SENDER_EMAIL
            password  = data.get("senderPass")     or SENDER_PASS
            recipient = data.get("recipientEmail") or RECIPIENT

            result = send_alert_email(data, sender, password, recipient)

            self.send_response(200)
            self._cors()
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())
        else:
            self.send_error(404)

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def log_message(self, fmt, *args):
        # Suppress static file logging, show only API calls
        if "/api/" in (args[0] if args else ""):
            print(f"[{datetime.now().strftime('%H:%M:%S')}] {fmt % args}")


# ---------- Entry point ----------
if __name__ == "__main__":
    server = HTTPServer(("0.0.0.0", SERVE_PORT), Handler)
    print("=" * 55)
    print("  Intelligent Brawl Monitor — Backend Server")
    print("=" * 55)
    print(f"  Serving at: http://localhost:{SERVE_PORT}")
    print()
    print("  Email config (set via environment variables):")
    print(f"    SENDER_EMAIL    = {SENDER_EMAIL or '(not set)'}")
    print(f"    SENDER_PASS     = {'(set)' if SENDER_PASS else '(not set)'}")
    print(f"    RECIPIENT_EMAIL = {RECIPIENT or '(not set)'}")
    print()
    print("  Or configure directly in the Settings tab of the UI.")
    print()
    print("  Press Ctrl+C to stop.")
    print("=" * 55)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
