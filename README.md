# SkyFlow Airlines Chatbot (NVIDIA + LangChain)

Console-based airline assistant that uses NVIDIA AI Endpoints (via LangChain) to chat with travelers and retrieve flight details from a CSV dataset.

## Author
- Name: Abdallah Salah  
- Email: abdallah.tech.ai@gmail.com

## Tech Stack
- Python 3.11+ (tested with 3.13)
- LangChain
- NVIDIA AI Endpoints (`ChatNVIDIA`)
- Pydantic v2
- CSV-based synthetic flight data (Kaggle)

## Project Structure
```
sky_flow_ai.py                  # main console chatbot
synthetic_flight_passenger_data.csv
requirements.txt
README.md
.env                            # not committed (API key, model name)
.venv/                          # local virtual env (ignored)
```
`.env` and `.venv` are local-only and should not be checked into Git.

## Prerequisites
- Python 3.11+ (3.13 recommended)
- Git
- NVIDIA AI Endpoints account and API key
- Basic familiarity with virtual environments

## Setup & Installation
1) Clone:
```bash
git clone https://github.com/<YOUR_USERNAME>/<YOUR_REPO_NAME>.git
cd <YOUR_REPO_NAME>
```
2) Create & activate virtualenv (PowerShell example):
```powershell
python -m venv .venv
.\.venv\Scripts\activate
```
3) Install dependencies:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```
4) Create `.env` in the project root:
```bash
NVIDIA_API_KEY=nvapi-your-real-key-here
MODEL_NAME=meta/llama-4-maverick-17b-l28e-instruct
```
5) Ensure `synthetic_flight_passenger_data.csv` sits next to `sky_flow_ai.py` (it is included in the repo).

## How to Run the Chatbot
```bash
python sky_flow_ai.py
```
You should see:
```
[ Agent ]: Hello! I'm your SkyFlow agent! How can I help you?
```
Then type your messages. Example:
```
[ Human ]: Hi, I'm Jane Doe. My passenger ID is P10.
[ Agent ]: Jane Doe's flight with confirmation P10, operated by <Airline>, departs from <Origin> to <Destination> at <Time>. The current status is On-time.
```
The bot streams responses in the console.

## How It Works (High-Level)
- Uses LangChain with `ChatNVIDIA` to generate assistant replies.
- Uses the same model (via `PydanticOutputParser`) to extract a `KnowledgeBase` (first_name, last_name, confirmation, discussion summary, open problems, current goals).
- `_load_flight_db` reads the CSV once at startup and builds `FLIGHT_DB` keyed by `Passenger_ID`.
- `get_user_info` treats the `confirmation` field as `Passenger_ID` (e.g., `P42`), looks up the record in `FLIGHT_DB`, and returns a natural-language flight summary.
- That summary is injected into the system prompt so the bot is aware of the passenger’s actual flight details.

### CSV schema (key columns)
`Passenger_ID, Flight_ID, Airline, Departure_Airport, Arrival_Airport, Departure_Time, Flight_Duration_Minutes, Flight_Status, Distance_Miles, Price_USD, Age, Gender, Income_Level, Travel_Purpose, Seat_Class, Bags_Checked, Frequent_Flyer_Status, Check_in_Method, Flight_Satisfaction_Score, Delay_Minutes, Booking_Days_In_Advance, No_Show, Weather_Impact, Seat_Selected`

## Customization
- Change `MODEL_NAME` in `.env` to any NVIDIA-supported chat model without code changes.
- Swap the CSV with another file (keep the same column names, or adjust the column mapping in `sky_flow_ai.py`).
- Extend the project into a REST API or web UI by wrapping the chat generator.

## Notes / Limitations
- Educational/demo project using synthetic data; no live booking systems are contacted.
- CSV is loaded once at startup; changes require restarting the script.
- Requires a valid NVIDIA API key with inference permissions for the chosen model.

## License
MIT License (for learning/testing; update as needed for production use).
