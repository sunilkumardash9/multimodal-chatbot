from typing import List, Union, ByteString
import gradio as gr
from gradio.components import Component
from pathlib import Path
from vertexai.preview.generative_models import GenerativeModel, Part, GenerationConfig



def add_content(prompt, chatbot, file):
    if file and prompt:
       chatbot = chatbot + [(prompt,None), ((file,),None)]
    elif prompt and not file:
        chatbot += [(prompt, None)]
    elif file and not prompt:
        chatbot += [((file,),None)]
    else:
        raise gr.Error("Enter a valid prompt or a file")
    return chatbot

def reset() -> List[Component]:
        return [
            gr.Dropdown(value = "gemini-pro",choices=["gemini-pro", "gemini-pro-vision"], label="Model", info="Choose a model", interactive=True),
            gr.Radio(label="Streaming", choices=[True, False], value=True, interactive=True, info="Stream while responses are generated"),
            gr.Slider(value= 0.6,maximum=1.0, label="Temparature", interactive=True),
            gr.Textbox(label="Token limit", value=2048),
            gr.Textbox(label="stop Sequence", info="Stops generation when the string is encountered."),
            gr.Slider(
                value=40, 
                label="Top-k",
                interactive=True,
                info="Top-k changes how the model selects tokens for output: lower = less random, higher = more diverse. Defaults to 40."
                ),
            gr.Slider(
                value=8, 
                label="Top-p",
                interactive=True,
                info="""Top-p changes how the model selects tokens for output. 
                        Tokens are selected from most probable to least until 
                        the sum of their probabilities equals the top-p value"""
                        )
            ]

class GeminiGenerator:
    """Multi-modal generator class for Gemini models"""

    def _convert_part(self, part: Union[str, ByteString, Part]) -> Part:
        if isinstance(part, str):
            return Part.from_text(part)
        elif isinstance(part, ByteString):
            return Part.from_data(part.data, part.mime_type)
        elif isinstance(part, Part):
            return part
        else:
            msg = f"Unsupported type {type(part)} for part {part}"
            raise ValueError(msg)
    
    def _check_file(self, file):
        if file:
            return True
        return False

    def run(self, history , prompt:str, file:str, model:str, stream: bool, temparature: float,
                  stop_sequence: str, top_k:int, top_p:float):
        generation_config = GenerationConfig(
            temperature=temparature,
            top_k=top_k,
            top_p=top_p,
            stop_sequences=stop_sequence
        )
        self.client = GenerativeModel(model_name = model, generation_config=generation_config,)
        if prompt and self._check_file(file):
            print('first')
            contents = [self._convert_part(part) for part in [prompt, file]]
            response = self.client.generate_content(contents=contents, stream=stream)
        elif prompt:
            print("sec")
            content = self._convert_part(prompt)
            response = self.client.generate_content(contents=content, stream=stream)
        elif self._check_file(file):
            print('third')
            content = self._convert_part(file)
            response = self.client.generate_content(contents=content, stream=stream)
        if stream:
            history[-1][-1] = ""
            for resp in response:
                history[-1][-1] += resp.candidates[0].content.parts[0].text
                yield "", gr.File(value=None), history
        else:
            history.append((None,response.candidates[0].content.parts[0].text))
            return " ", gr.File(value=None), history

gemini_generator = GeminiGenerator()

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale=3):
            gr.Markdown("Chatbot")
            chatbot = gr.Chatbot(show_copy_button=True, height=650)
            with gr.Row():
                with gr.Column(scale=6):
                    prompt = gr.Textbox(placeholder="write prompt")
                with gr.Column(scale=2):
                    file = gr.File()
                with gr.Column(scale=2):
                    button = gr.Button()
        with gr.Column(scale=1):
            with gr.Row():
                gr.Markdown("Model Parameters")
                reset_params = gr.Button(value="Reset")
            gr.Markdown("General")
            model = gr.Dropdown(value = "gemini-pro-vision",choices=["gemini-pro", "gemini-pro-vision"], label="Model", info="Choose a model", interactive=True)
            stream = gr.Radio(label="Streaming", choices=[True, False], value=True, interactive=True, info="Stream while responses are generated")
            temparature = gr.Slider(value= 0.6,maximum=1.0, label="Temparature", interactive=True)
            stop_sequence = gr.Textbox(label="stop Sequence", info="Stops generation when the string is encountered.")
            gr.Markdown(value="Advanced")
            top_p = gr.Slider(
                value=0.4,
                maximum=1.0,
                label="Top-k",
                interactive=True,
                info="Top-k changes how the model selects tokens for output: lower = less random, higher = more diverse. Defaults to 0.4."
                )
            top_k = gr.Slider(
                value=8, 
                label="Top-p",
                interactive=True,
                info="""Top-p changes how the model selects tokens for output. 
                        Tokens are selected from most probable to least until 
                        the sum of their probabilities equals the top-p value"""
                        )
    
    button.click(fn=add_content, inputs=[prompt, chatbot, file], outputs=[chatbot])\
    .success(fn = gemini_generator.run, inputs=[chatbot, prompt, file, model, stream, temparature, stop_sequence,
                                     top_k, top_p], outputs=[prompt,file,chatbot]
                                     )
    reset_params.click(fn=reset, outputs=[model, stream, temparature,
                                          stop_sequence, top_p, top_k])


demo.queue()
demo.launch()
            

            
