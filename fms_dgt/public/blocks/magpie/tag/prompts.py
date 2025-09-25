# Standard
from typing import List

# Third Party
from jinja2 import Template

# Local
from fms_dgt.base.prompt import PromptTemplate
from fms_dgt.utils import read_file


class MagpieTransformPrompt(PromptTemplate):
    """
    Prompt generator for Magpie
    """

    def __init__(
        self,
        template_path: str,
        prompt_type: str,
        stop: List[str] = None,
    ):
        prompt_template = read_file(file_path=template_path)
        prompt_template = prompt_template.replace("{{", "{% raw %}{{").replace(
            "}}", "}}{% endraw %}"
        )

        self._template = Template(prompt_template).render()
        self._prompt_type = prompt_type
        self._stop = stop if stop else ["</s>"]

    @property
    def stop(self) -> List[str]:
        return self._stop

    def encode(
        self,
        input: str,
        output: str = None,
    ):
        def apply_conversation_template(utterances):
            str_conversations = ""
            label = ""
            turn_user = 1
            turn_resp = 1
            for utterance in utterances:
                if "from" in utterance:
                    typ_field = "from"
                elif "role" in utterance:
                    typ_field = "role"
                elif "speaker" in utterance:
                    typ_field = "speaker"
                else:
                    raise ValueError(
                        "conversation should have a 'from' field or a 'role' field or a 'speaker' field to signify whether it was a user or assistant utterance"
                    )

                if "text" in utterance:
                    txt_field = "text"
                elif "value" in utterance:
                    txt_field = "value"
                else:
                    txt_field = "content"

                if utterance[typ_field] == "user":
                    label = f"### User Question: {str(turn_user)}"
                    turn_user += 1
                elif utterance[typ_field] == "assistant":
                    if txt_field not in utterance:
                        # Skipping messages without content
                        continue
                    else:
                        label = f"### Assistant Response: {str(turn_resp)}"
                        turn_resp += 1
                else:
                    # Skipping roles that are not 'user' or 'assistant'
                    continue

                text = f"""\n{label}\n{utterance[txt_field]}\n"""
                str_conversations += text

            return str_conversations

        if self._prompt_type == "multi_turn":
            str_conversation = apply_conversation_template(input)

            render_dict = {"str_conversation": str_conversation}
        else:
            render_dict = {"input": input, "output": output}

        return self._template.format(**render_dict)
