from django import forms

class ContactForm(forms.Form):
    name = forms.CharField(max_length=100, label='Nombre')
    email = forms.EmailField(label='Correo Electrónico')
    message = forms.CharField(widget=forms.Textarea, label='Mensaje')
