# Betfair price-capture agent

This small program sends live Betfair prices to the greyhound app. It has
to run on a computer **in Ireland or the UK**, because Betfair blocks
connections from the country the app itself is hosted in.

Your Betfair username and password stay on this computer. They are never
sent to the app or to anyone else. The only thing shared with the app is
prices, using a token that can do nothing except send prices.

## What you need

- A computer in Ireland or the UK that can be on during racing hours
- Python 3 (Windows: get it from python.org and tick "Add Python to PATH"
  during install; macOS and Linux already have it)
- The Betfair delayed application key, plus the Betfair login

## Setup (once)

1. Put the `agent` folder somewhere easy to find, e.g. your Desktop.
2. Make a copy of `agent.env.example` and name the copy `agent.env`.
3. Open `agent.env` in any text editor and fill in the five values.
   `INGEST_TOKEN` comes with these instructions.
4. Open a terminal (macOS: Terminal; Windows: Command Prompt) in that
   folder and check everything works:

   ```
   python3 betfair_capture_agent.py --check
   ```

   Success looks like a list of upcoming races and "Config looks good."
   Nothing is sent to the app in check mode.

## Running it

```
python3 betfair_capture_agent.py
```

Leave the window open. It checks for prices every 20 minutes between
11:00 and 23:00 and prints one line per check. Press Ctrl-C to stop.

That is all it does — it reads prices and forwards them. **It never
places a bet and it cannot place a bet.**

## Leaving it running permanently

A laptop that sleeps will miss races. Two options:

- **Raspberry Pi** (about €60) — leave it plugged in, and it runs
  every day without touching your computer.
- **Scheduled task** — run `python3 betfair_capture_agent.py --once`
  every 20 minutes via cron (macOS/Linux) or Task Scheduler (Windows),
  instead of leaving the window open.

## If something goes wrong

| Message | What it means |
|---|---|
| `403 Forbidden` | The connection is being blocked. Check you are not on a VPN routing through another country. |
| `Betfair login rejected: SUSPENDED` | The login worked — the Betfair account itself needs attention. Log in at betfair.com in a browser; for a new account this is usually identity verification (photo ID and proof of address). |
| `Betfair login rejected: INVALID_USERNAME_OR_PASSWORD` | Wrong details. Betfair wants the username, not the email address. |
| `Betfair login rejected: <anything else>` | The message now explains the specific code and what to do. |
| `INVALID_APP_KEY` | The app key does not belong to the account you logged in with. If you switched Betfair accounts, create a new app key from the NEW account. |
| `Check FAILED` | The check found a real problem — the message above it says what. Nothing was sent. |
| `no Irish markets in the next 120 min` | Normal outside racing hours, or when no Irish meetings are on. |
| `could not reach the app` | Internet problem, or the app is restarting. It will retry on the next check. |
| `unmatched: ...` | Prices arrived but the app could not match that track to a race. Worth reporting — usually a track-name spelling difference. |
