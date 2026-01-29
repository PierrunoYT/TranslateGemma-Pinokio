import gradio as gr
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, pipeline
from PIL import Image
import os
import requests
from io import BytesIO
from huggingface_hub import login

# Global variables to store model and processor
model = None
processor = None
pipe = None
current_model_size = None
hf_token_set = False

# Supported languages mapping (55 main languages)
LANGUAGES = {
    "Arabic": "ar",
    "Bengali": "bn",
    "Bulgarian": "bg",
    "Catalan": "ca",
    "Chinese (Simplified)": "zh",
    "Chinese (Traditional)": "zh-TW",
    "Croatian": "hr",
    "Czech": "cs",
    "Danish": "da",
    "Dutch": "nl",
    "English": "en",
    "English (US)": "en-US",
    "English (UK)": "en-GB",
    "Estonian": "et",
    "Finnish": "fi",
    "French": "fr",
    "French (Canada)": "fr-CA",
    "German": "de",
    "German (Austria)": "de-AT",
    "German (Switzerland)": "de-CH",
    "Greek": "el",
    "Gujarati": "gu",
    "Hebrew": "he",
    "Hindi": "hi",
    "Hungarian": "hu",
    "Icelandic": "is",
    "Indonesian": "id",
    "Italian": "it",
    "Japanese": "ja",
    "Kannada": "kn",
    "Korean": "ko",
    "Latvian": "lv",
    "Lithuanian": "lt",
    "Macedonian": "mk",
    "Malayalam": "ml",
    "Marathi": "mr",
    "Norwegian": "no",
    "Persian": "fa",
    "Polish": "pl",
    "Portuguese": "pt",
    "Portuguese (Brazil)": "pt-BR",
    "Portuguese (Portugal)": "pt-PT",
    "Punjabi": "pa",
    "Romanian": "ro",
    "Russian": "ru",
    "Serbian": "sr",
    "Slovak": "sk",
    "Slovenian": "sl",
    "Spanish": "es",
    "Spanish (Mexico)": "es-MX",
    "Spanish (Spain)": "es-ES",
    "Swedish": "sv",
    "Tamil": "ta",
    "Telugu": "te",
    "Thai": "th",
    "Turkish": "tr",
    "Ukrainian": "uk",
    "Urdu": "ur",
    "Vietnamese": "vi",
}

def set_hf_token(token):
    """Set Hugging Face token for authentication"""
    global hf_token_set
    
    if not token or not token.strip():
        return "⚠️ Please enter a valid Hugging Face token"
    
    try:
        login(token=token.strip(), add_to_git_credential=False)
        hf_token_set = True
        return "✓ Hugging Face token set successfully! You can now load models."
    except Exception as e:
        return f"❌ Error setting token: {str(e)}"


def load_model(model_size="12B", use_pipeline=True):
    """Load the TranslateGemma model"""
    global model, processor, pipe, current_model_size, hf_token_set
    
    # Check if token is set (or if already logged in via CLI)
    if not hf_token_set:
        try:
            # Try to load without explicit token (in case user logged in via CLI)
            pass
        except:
            pass
    
    # If already loaded and same size, skip
    if current_model_size == model_size and (pipe is not None or model is not None):
        return f"Model {model_size} already loaded ✓"
    
    # Clear existing model
    if model is not None:
        del model
        del processor
        model = None
        processor = None
    if pipe is not None:
        del pipe
        pipe = None
    
    torch.cuda.empty_cache()
    
    model_id = f"google/translategemma-{model_size.lower()}-it"
    
    try:
        if use_pipeline:
            pipe = pipeline(
                "image-text-to-text",
                model=model_id,
                device="cuda" if torch.cuda.is_available() else "cpu",
                dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
            )
            current_model_size = model_size
            return f"✓ Model {model_size} loaded successfully using pipeline (CUDA: {torch.cuda.is_available()})"
        else:
            processor = AutoProcessor.from_pretrained(model_id)
            model = AutoModelForImageTextToText.from_pretrained(
                model_id,
                device_map="auto",
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
            )
            current_model_size = model_size
            return f"✓ Model {model_size} loaded successfully (CUDA: {torch.cuda.is_available()})"
    except Exception as e:
        error_msg = str(e)
        if "401" in error_msg or "authentication" in error_msg.lower():
            return f"❌ Authentication error. Please:\n1. Enter your Hugging Face token above\n2. Accept the license at: https://huggingface.co/{model_id}"
        return f"❌ Error loading model: {error_msg}\n\nMake sure you have accepted the license at: https://huggingface.co/{model_id}"


def translate_text(text, source_lang, target_lang, max_tokens=200):
    """Translate text from source to target language"""
    global pipe, model, processor
    
    if not text or not text.strip():
        return "⚠️ Please enter text to translate"
    
    if pipe is None and model is None:
        return "⚠️ Please load a model first using the 'Load Model' button"
    
    source_code = LANGUAGES.get(source_lang, "en")
    target_code = LANGUAGES.get(target_lang, "es")
    
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "source_lang_code": source_code,
                    "target_lang_code": target_code,
                    "text": text
                }
            ]
        }
    ]
    
    try:
        if pipe is not None:
            output = pipe(text=messages, max_new_tokens=max_tokens)
            return output[0]["generated_text"][-1]["content"]
        else:
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            ).to(model.device, dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32)
            
            input_len = len(inputs['input_ids'][0])
            
            with torch.inference_mode():
                generation = model.generate(**inputs, do_sample=False, max_new_tokens=max_tokens)
                generation = generation[0][input_len:]
                decoded = processor.decode(generation, skip_special_tokens=True)
            
            return decoded
    except Exception as e:
        return f"❌ Translation error: {str(e)}"


def translate_image(image, source_lang, target_lang, max_tokens=200):
    """Extract and translate text from image"""
    global pipe, model, processor
    
    if image is None:
        return "⚠️ Please upload an image"
    
    if pipe is None and model is None:
        return "⚠️ Please load a model first using the 'Load Model' button"
    
    source_code = LANGUAGES.get(source_lang, "en")
    target_code = LANGUAGES.get(target_lang, "es")
    
    try:
        # Save image temporarily
        temp_path = "temp_image.jpg"
        if isinstance(image, str):
            # If image is a URL
            response = requests.get(image)
            img = Image.open(BytesIO(response.content))
            img.save(temp_path)
        else:
            # If image is a PIL Image or numpy array
            if not isinstance(image, Image.Image):
                image = Image.fromarray(image)
            image.save(temp_path)
        
        # Create URL-like path for the image
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source_lang_code": source_code,
                        "target_lang_code": target_code,
                        "url": temp_path
                    }
                ]
            }
        ]
        
        if pipe is not None:
            output = pipe(text=messages, max_new_tokens=max_tokens)
            result = output[0]["generated_text"][-1]["content"]
        else:
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            ).to(model.device, dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32)
            
            input_len = len(inputs['input_ids'][0])
            
            with torch.inference_mode():
                generation = model.generate(**inputs, do_sample=False, max_new_tokens=max_tokens)
                generation = generation[0][input_len:]
                result = processor.decode(generation, skip_special_tokens=True)
        
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        return result
    except Exception as e:
        return f"❌ Image translation error: {str(e)}"


# Create Gradio interface
with gr.Blocks(title="TranslateGemma - Multilingual Translation") as demo:
    gr.Markdown(
        """
        # 🌍 TranslateGemma - AI Translation
        
        **Google's open-source translation model supporting 55+ languages**
        
        - 🔄 Text translation across 55 languages
        - 🖼️ Extract and translate text from images
        - ⚡ Powered by Gemma 3 architecture
        """
    )
    
    # Hugging Face Token Section
    with gr.Accordion("🔑 Hugging Face Authentication", open=True):
        gr.Markdown(
            """
            **First time setup:**
            1. Create account at [huggingface.co](https://huggingface.co)
            2. Accept license at [huggingface.co/google/translategemma-12b-it](https://huggingface.co/google/translategemma-12b-it)
            3. Create a Read token in Settings → Access Tokens
            4. Paste your token below
            
            *Skip this if you've already logged in via `huggingface-cli login`*
            """
        )
        with gr.Row():
            hf_token_input = gr.Textbox(
                label="Hugging Face Token",
                placeholder="hf_...",
                type="password",
                scale=3
            )
            token_btn = gr.Button("Set Token", variant="secondary", scale=1)
        token_status = gr.Textbox(
            label="Token Status",
            value="Token not set (optional if already logged in via CLI)",
            interactive=False
        )
    
    # Model Loading Section
    with gr.Row():
        with gr.Column(scale=1):
            model_size = gr.Radio(
                choices=["4B", "12B", "27B"],
                value="12B",
                label="Model Size",
                info="4B: Fast, mobile | 12B: Balanced (recommended) | 27B: Best quality"
            )
            load_btn = gr.Button("🚀 Load Model", variant="primary", size="lg")
            model_status = gr.Textbox(
                label="Model Status",
                value="Set token (if needed) and click 'Load Model' to start",
                interactive=False
            )
    
    gr.Markdown("---")
    
    with gr.Tabs() as tabs:
        # Text Translation Tab
        with gr.Tab("📝 Text Translation"):
            with gr.Row():
                with gr.Column():
                    text_input = gr.Textbox(
                        label="Enter text to translate",
                        placeholder="Type or paste your text here...",
                        lines=8
                    )
                    with gr.Row():
                        text_source_lang = gr.Dropdown(
                            choices=list(LANGUAGES.keys()),
                            value="English",
                            label="Source Language"
                        )
                        text_target_lang = gr.Dropdown(
                            choices=list(LANGUAGES.keys()),
                            value="Spanish",
                            label="Target Language"
                        )
                    text_max_tokens = gr.Slider(
                        minimum=50,
                        maximum=500,
                        value=200,
                        step=10,
                        label="Max Output Tokens"
                    )
                    text_translate_btn = gr.Button("🔄 Translate", variant="primary", size="lg")
                
                with gr.Column():
                    text_output = gr.Textbox(
                        label="Translation",
                        placeholder="Translation will appear here...",
                        lines=8,
                        interactive=False
                    )
            
            # Example texts
            gr.Examples(
                examples=[
                    ["Hello, how are you today?", "English", "Spanish"],
                    ["Bonjour, comment allez-vous?", "French", "English"],
                    ["こんにちは、元気ですか？", "Japanese", "English"],
                    ["Hola, ¿cómo estás?", "Spanish", "French"],
                    ["Guten Tag, wie geht es Ihnen?", "German", "Italian"],
                ],
                inputs=[text_input, text_source_lang, text_target_lang],
                label="Example Translations"
            )
        
        # Image Translation Tab
        with gr.Tab("🖼️ Image Translation"):
            with gr.Row():
                with gr.Column():
                    image_input = gr.Image(
                        label="Upload image with text",
                        type="pil"
                    )
                    with gr.Row():
                        image_source_lang = gr.Dropdown(
                            choices=list(LANGUAGES.keys()),
                            value="English",
                            label="Source Language"
                        )
                        image_target_lang = gr.Dropdown(
                            choices=list(LANGUAGES.keys()),
                            value="Spanish",
                            label="Target Language"
                        )
                    image_max_tokens = gr.Slider(
                        minimum=50,
                        maximum=500,
                        value=200,
                        step=10,
                        label="Max Output Tokens"
                    )
                    image_translate_btn = gr.Button("🔄 Extract & Translate", variant="primary", size="lg")
                
                with gr.Column():
                    image_output = gr.Textbox(
                        label="Extracted & Translated Text",
                        placeholder="Extracted and translated text will appear here...",
                        lines=10,
                        interactive=False
                    )
        
        # About Tab
        with gr.Tab("ℹ️ About"):
            gr.Markdown(
                """
                ## About TranslateGemma
                
                TranslateGemma is Google's family of open-source translation models released in January 2026.
                Built on the Gemma 3 architecture, these models deliver state-of-the-art translation quality.
                
                ### Key Features:
                - **55 Languages**: Supports major world languages including English, Spanish, French, German, Chinese, Japanese, Arabic, Hindi, and more
                - **3 Model Sizes**: 
                  - 4B: Optimized for mobile and edge devices
                  - 12B: Balanced performance for laptops (recommended)
                  - 27B: Highest quality for cloud deployment
                - **Multimodal**: Can extract and translate text from images
                - **High Performance**: 26% better accuracy than base models, 30% improvement on rare language pairs
                
                ### Requirements:
                - **GPU**: CUDA-compatible GPU recommended (CPU supported but slower)
                - **VRAM**: 4B (~8GB), 12B (~16GB), 27B (~32GB)
                - **Hugging Face Account**: Required to accept model license
                
                ### Resources:
                - [Hugging Face Models](https://huggingface.co/collections/google/translategemma)
                - [Technical Paper](https://arxiv.org/pdf/2601.09012)
                - [Google Blog](https://blog.google/technology/ai/translategemma/)
                
                ### License:
                Google Gemma License - Free for research and commercial use
                """
            )
    
    # Event handlers
    token_btn.click(
        fn=set_hf_token,
        inputs=[hf_token_input],
        outputs=[token_status]
    )
    
    load_btn.click(
        fn=load_model,
        inputs=[model_size],
        outputs=[model_status]
    )
    
    text_translate_btn.click(
        fn=translate_text,
        inputs=[text_input, text_source_lang, text_target_lang, text_max_tokens],
        outputs=[text_output]
    )
    
    image_translate_btn.click(
        fn=translate_image,
        inputs=[image_input, image_source_lang, image_target_lang, image_max_tokens],
        outputs=[image_output]
    )

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True
    )
