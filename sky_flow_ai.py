import os
from typing import Optional
from operator import itemgetter
import csv
from pathlib import Path

# Load .env
from dotenv import load_dotenv
load_dotenv()


def _get_api_key() -> str:
    key = os.getenv("NVIDIA_API_KEY") or os.getenv("NVCF_API_KEY") or ""
    if not key:
        raise RuntimeError("NVIDIA_API_KEY (or NVCF_API_KEY) is missing from .env")
    print(f"[AUTH] NVIDIA API key loaded (len={len(key)}).")
    return key


def _get_model_name() -> str:
    model = (os.getenv("MODEL_NAME") or "").strip()
    if not model:
        raise RuntimeError("MODEL_NAME is missing from .env (e.g., nvidia/llama-3.1-nemotron-70b-instruct)")
    print(f"[INIT] Using model: {model}")
    return model


API_KEY = _get_api_key()
MODEL_NAME = _get_model_name()

# LangChain / NVIDIA
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
from langchain.schema.runnable.passthrough import RunnableAssign
from langchain.schema.runnable import RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.output_parsers import StrOutputParser


class KnowledgeBase(BaseModel):
    first_name: str = Field("unknown", description="User first name, 'unknown' if unknown")
    last_name: str = Field("unknown", description="User last name, 'unknown' if unknown")
    # Passenger_ID مثل P42 لذلك نخليه نصي
    confirmation: Optional[str] = Field(
        None,
        description="Passenger / booking ID (e.g. 'P42'); None if unknown",
    )
    discussion_summary: str = Field("", description="Running summary of the conversation")
    open_problems: str = Field("no problems till now", description="Still unresolved topics")
    current_goals: str = Field("", description="Current user goal")


# ---------- Flight "database" loaded from synthetic_flight_passenger_data.csv ----------

BASE_DIR = Path(__file__).resolve().parent
# تأكد أن اسم الملف يطابق اسم ملف الـ CSV عندك
FLIGHTS_CSV_PATH = BASE_DIR / "synthetic_flight_passenger_data.csv"

# هذا هو عمود رقم الحجز في الملف
PASSENGER_ID_COL = "Passenger_ID"


def _load_flight_db(csv_path: Path) -> dict[str, dict]:
    """
    Load flight records from the Kaggle CSV into a dict keyed by Passenger_ID.
    Keys are uppercased (P1, P2, ...) to allow case-insensitive lookup.
    """
    db: dict[str, dict] = {}

    if not csv_path.exists():
        print(f"[WARN] Flights DB file not found: {csv_path}")
        return db

    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise RuntimeError(f"{csv_path.name} has no header row.")

        # تأكد أن Passenger_ID موجود في الهيدر
        if PASSENGER_ID_COL not in reader.fieldnames:
            raise RuntimeError(
                f"Expected column '{PASSENGER_ID_COL}' in {csv_path.name}, "
                f"but found: {reader.fieldnames}"
            )

        for row in reader:
            raw_id = row.get(PASSENGER_ID_COL, "")
            pid = str(raw_id).strip().upper()
            if not pid:
                continue
            db[pid] = row

    print(f"[INIT] Loaded {len(db)} flights from {csv_path.name}. Using ID column: {PASSENGER_ID_COL}")
    return db


FLIGHT_DB = _load_flight_db(FLIGHTS_CSV_PATH)


def RExtract(pydantic_class: type[BaseModel], llm, prompt) -> RunnableLambda:
    parser = PydanticOutputParser(pydantic_object=pydantic_class)
    instruct_merge = RunnableAssign({"format_instructions": lambda x: parser.get_format_instructions()})

    def preparse(user_input: str) -> str:
        if "{" not in user_input:
            user_input = "{" + user_input
        if "}" not in user_input:
            user_input += "}"
        return (
            user_input.replace("\\_", "_")
            .replace("\n", " ")
            .replace("\\]", "]")
            .replace("\\[", "[")
        )

    return instruct_merge | prompt | llm | preparse | parser


def get_key_fn(base: KnowledgeBase) -> dict:
    return {
        "first_name": base.first_name,
        "last_name": base.last_name,
        "confirmation": base.confirmation,
    }


get_key = RunnableLambda(get_key_fn)


def get_user_info(user_data: dict) -> str:
    """
    Lookup flight info from FLIGHT_DB based on confirmation (Passenger_ID).
    We treat KnowledgeBase.confirmation as the Passenger_ID from the CSV (P1, P2, ...).
    """
    req_keys = ["first_name", "last_name", "confirmation"]
    assert all((k in user_data) for k in req_keys), f"Expected keys {req_keys}, got {user_data}"

    if not FLIGHT_DB:
        return (
            "I could not access the flights database right now. "
            "Please try again later or contact support."
        )

    conf_raw = user_data.get("confirmation")
    # نتعامل معها case-insensitive: p42 -> P42
    conf = str(conf_raw).strip().upper() if conf_raw is not None else ""

    if not conf:
        return (
            "I do not have a confirmation number for this passenger yet. "
            "If they need flight details, ask them for their booking confirmation "
            "(for example, their passenger ID like 'P42')."
        )

    record = FLIGHT_DB.get(conf)
    if not record:
        return (
            f"No flight information was found for confirmation '{conf}'. "
            "If it's important, ask them to double-check their confirmation number."
        )

    # أعمدة الملف كما في الهيدر:
    # Passenger_ID,Flight_ID,Airline,Departure_Airport,Arrival_Airport,
    # Departure_Time,Flight_Duration_Minutes,Flight_Status,Distance_Miles,
    # Price_USD,Age,Gender,Income_Level,Travel_Purpose,Seat_Class,Bags_Checked,
    # Frequent_Flyer_Status,Check_in_Method,Flight_Satisfaction_Score,Delay_Minutes,
    # Booking_Days_In_Advance,No_Show,Weather_Impact,Seat_Selected

    airline = record.get("Airline", "Unknown airline")
    dep_airport = record.get("Departure_Airport", "unknown departure airport")
    arr_airport = record.get("Arrival_Airport", "unknown arrival airport")
    dep_time = record.get("Departure_Time", "an unknown time")
    status = str(record.get("Flight_Status", "")).strip()
    seat_class = record.get("Seat_Class", "")
    price = record.get("Price_USD", "")
    seat_selected = record.get("Seat_Selected", "")
    delay_minutes = record.get("Delay_Minutes", "")
    check_in_method = record.get("Check_in_Method", "")
    ff_status = record.get("Frequent_Flyer_Status", "")

    status_part = f" The current status is {status}." if status else ""
    seat_part = f" Seat class: {seat_class}." if seat_class else ""
    price_part = f" Ticket price: ${price}." if price not in ("", None) else ""
    delay_part = (
        f" Reported delay: {delay_minutes} minutes." if str(delay_minutes).strip() not in ("", "0", "0.0") else ""
    )
    check_in_part = f" Check-in method: {check_in_method}." if check_in_method else ""
    ff_part = f" Frequent flyer status: {ff_status}." if ff_status else ""

    # الاسم نأخذه من الـ KnowledgeBase فقط (لأن الملف لا يحتوي على أعمدة اسم)
    kb_first = user_data.get("first_name") or "Passenger"
    kb_last = user_data.get("last_name") or ""
    full_name = f"{kb_first} {kb_last}".strip()

    return (
        f"{full_name}'s flight with confirmation {conf}, operated by {airline}, "
        f"departs from {dep_airport} to {arr_airport} at {dep_time}.{status_part}"
        f"{seat_part}{price_part}{delay_part}{check_in_part}{ff_part}"
    )


external_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a chatbot for SkyFlow Airlines, and you are helping a customer with their issue. "
            "Please chat with them! Stay concise and clear! "
            "Your running knowledge base is: {know_base}. This is for you only; do not mention it! "
            "\nUsing that, we retrieved the following: {context}\n"
            "If they provide info and the retrieval fails, ask to confirm their first/last name and confirmation. "
            "Do not ask any other personal info. If it's not important, don't ask. "
            "The checking happens automatically; you cannot check manually.",
        ),
        ("assistant", "{output}"),
        ("user", "{input}"),
    ]
)

parser_prompt = ChatPromptTemplate.from_template(
    "You are a chat assistant representing the airline SkyFlow, tracking info about the conversation. "
    "You have just received a message from the user. Please fill in the schema based on the chat.\n\n"
    "{format_instructions}\n\n"
    "Use the knowledge base context below and keep fields unchanged when information is missing.\n\n"
    "OLD KNOWLEDGE BASE: {know_base}\n\n"
    "ASSISTANT RESPONSE: {output}\n\n"
    "USER MESSAGE: {input}\n\n"
    "NEW KNOWLEDGE BASE:"
)


def make_llm(model: str, temperature: float = 0.2):
    return ChatNVIDIA(model=model, api_key=API_KEY, temperature=temperature)


chat_llm = make_llm(MODEL_NAME) | StrOutputParser()
instruct_llm = make_llm(MODEL_NAME) | StrOutputParser()

knowbase_getter = RExtract(KnowledgeBase, instruct_llm, parser_prompt)
database_getter = itemgetter("know_base") | get_key | get_user_info
external_chain = external_prompt | chat_llm

internal_chain = (
    RunnableAssign({"know_base": knowbase_getter})
    | RunnableAssign({"context": database_getter})
)

state = {"know_base": KnowledgeBase()}


def chat_gen(message: str, history=None, return_buffer=True):
    global state
    history = history or []

    state["input"] = message
    state["history"] = history
    state["output"] = "" if not history else (history[-1][1] or "")

    state = internal_chain.invoke(state)

    buffer = ""
    for token in external_chain.stream(state):
        buffer += token
        yield buffer if return_buffer else token


def run_console(max_turns=100):
    chat_history = [[None, "Hello! I'm your SkyFlow agent! How can I help you?"]]
    print("\n[ Agent ]:", chat_history[0][1])

    for _ in range(max_turns):
        try:
            message = input("\n[ Human ]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye!")
            break

        if not message:
            continue
        if message.lower() in {"/exit", "exit", ":q"}:
            print("bye!")
            break

        print("\n[ Agent ]: ")
        history_entry = [message, ""]
        for token in chat_gen(message, chat_history, return_buffer=False):
            print(token, end="")
            history_entry[1] += token
        chat_history.append(history_entry)
        print("\n")


if __name__ == "__main__":
    run_console()
