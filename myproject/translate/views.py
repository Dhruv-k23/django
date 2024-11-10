from django.shortcuts import render
from transformers import MarianMTModel, MarianTokenizer
from django.http import HttpResponse
model_name = 'Helsinki-NLP/opus-mt-hi-en'
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

def translate_text(request):
    translated_text = ""
    if request.method == "POST":
        input_text = request.POST.get("input_text")
        if input_text:
            # Translation function
            inputs = tokenizer(input_text, return_tensors="pt", padding=True)
            translated = model.generate(**inputs)
            translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
    return render(request, 'translate/index.html', {'translated_text': translated_text})

def home(request):
    return HttpResponse("hello!")

