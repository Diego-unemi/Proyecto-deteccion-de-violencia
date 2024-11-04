from django import forms
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from .models import CustomUser

class ContactForm(forms.Form):
    name = forms.CharField(max_length=100, label='Nombre')
    email = forms.EmailField(label='Correo Electrónico')
    message = forms.CharField(widget=forms.Textarea, label='Mensaje')

class VideoUploadForm(forms.Form):
    video = forms.FileField(label="Subir Video")

class CustomUserCreationForm(UserCreationForm):
    email = forms.EmailField(required=True, widget=forms.EmailInput(attrs={
        'class': 'w-full px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-600',
        'placeholder': 'Ingresa tu correo'
    }))
    phone_number = forms.CharField(max_length=13, widget=forms.TextInput(attrs={
        'class': 'w-full px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-600',
        'placeholder': 'Ingresa tu número de celular (+593)'
    }))

    class Meta:
        model = CustomUser
        fields = ['username', 'email', 'phone_number', 'password1', 'password2']
        widgets = {
            'username': forms.TextInput(attrs={
                'class': 'w-full px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-600',
                'placeholder': 'Elige un nombre de usuario'
            }),
            'password1': forms.PasswordInput(attrs={
                'class': 'w-full px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-600',
                'placeholder': 'Crea una contraseña'
            }),
            'password2': forms.PasswordInput(attrs={
                'class': 'w-full px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-600',
                'placeholder': 'Repite la contraseña'
            }),
        }

    def clean_phone_number(self):
        phone = self.cleaned_data.get('phone_number')
        # Verifica que el número comience con '+593' y tenga 13 caracteres (incluido el '+')
        if not phone.startswith('+593') or len(phone) != 13:
            raise forms.ValidationError("Ingrese un número de celular ecuatoriano válido en formato +593xxxxxxxxx.")
        return phone

class CustomAuthenticationForm(AuthenticationForm):
    username = forms.CharField(max_length=150)
    password = forms.CharField(widget=forms.PasswordInput(attrs={
        'class': 'w-full px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-600',
        'placeholder': 'Ingresa tu contraseña'
    }))

class VideoUploadForm(forms.Form):
    video_file = forms.FileField(label='Seleccionar archivo de video')
    video_name = forms.CharField(max_length=100, label='Nombre del video', required=False)
    video_description = forms.CharField(widget=forms.Textarea, required=False, label='Descripción (opcional)')

