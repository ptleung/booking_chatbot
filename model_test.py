from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

model_name = "deepset/roberta-large-squad2"

nlp = pipeline("question-answering", model=model_name, tokenizer=model_name)

context1 = "Hello, I would like to book studio95+65 on 19/8 from 9 to 11 am."
context2 = "Hello, I would like to book studio35 next thursday 10pm for 2 hours."

question_room = "what is the room they want to book?"
question_date = "which date do they want to book?"
question_time = "which clock do they wish to start the booking?"
question_time_span = "how long do they want to book the room?"


def generate_question_set(question, context):
    question_set = {
        "question": question,
        "context": context
    }
    return question_set


def generate_log_statement(output_json, question_type):
    print(f"{question_type}: {output_json['answer']}, "
          f"with {round(output_json['score'], 3)*100} confidence.")


def extract_booking_question(context):
    ans_room = nlp(generate_question_set(question_room, context))
    ans_date = nlp(generate_question_set(question_date, context))
    ans_time = nlp(generate_question_set(question_time, context))
    ans_time_span = nlp(generate_question_set(question_time_span, context))
    generate_log_statement(ans_room, "room")
    generate_log_statement(ans_date, "date")
    generate_log_statement(ans_time, "time")
    if ans_time['answer'] != ans_time_span['answer']:
        generate_log_statement(ans_time_span, "duration")


while True:
    _context = input()
    if _context == "quit":
        break
    extract_booking_question(_context)
    print("\n")