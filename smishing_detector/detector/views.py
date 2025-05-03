from django.shortcuts import render
from .models import Message
from .utils import predict_smishing
from django.views.decorators.csrf import csrf_exempt
from django.http import HttpResponse
from twilio.twiml.messaging_response import MessagingResponse


def index(request):
    if request.method == "POST":
        text = request.POST.get("text")
        result = predict_smishing(text)
        Message.objects.create(text=text, result=result)
    messages = Message.objects.all().order_by("-created_at")
    return render(request, "detector/index.html", {"messages": messages})


@csrf_exempt
def whatsapp_webhook(request):
    if request.method == 'POST':
        from_number = request.POST.get('From')
        body = request.POST.get('Body')

        result = predict_smishing(body)
        print(f"Message from {from_number}: {body} → {result}")

        # Optional: respond back via Twilio
        response_msg = "⚠️ This message looks suspicious!" if result == 'smishing' else "✅ This message looks safe."
        twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
        <Response>
            <Message>{response_msg}</Message>
        </Response>"""
        return HttpResponse(twiml, content_type='text/xml')

    return HttpResponse("Only POST requests are allowed.")