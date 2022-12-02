from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class chatbot:

    def __init__(self, num_info_questions=1, previous_dialog=None) -> None:
        self.n_info_qs = num_info_questions
        self.state = 0
        self.instruction = f"Instruction: given some dialog, learn about the user's reading preferences and recommend a book."

        if previous_dialog is None:
            self.dialog = []
        else:
            self.dialog = previous_dialog

        self.tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/GODEL-v1_1-large-seq2seq")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            "microsoft/GODEL-v1_1-large-seq2seq")

    def generate_response(self, knowledge, input_text, instruction=None):
        if knowledge != '':
            knowledge = '[KNOWLEDGE] ' + knowledge

        if instruction is None:
            instruction = self.instruction

        self.dialog.append(input_text)
        dialog = ' EOS '.join(self.dialog)

        query = f"{instruction} [CONTEXT] {dialog} {knowledge}"
        input_ids = self.tokenizer(f"{query}", return_tensors="pt").input_ids
        outputs = self.model.generate(input_ids, max_length=128,
                                      min_length=8, top_p=0.9, do_sample=True)
        output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        self.dialog.append(output)
        return output
