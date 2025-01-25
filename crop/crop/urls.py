from django.contrib import admin
from django.urls import path, include  # include ইমপোর্ট করুন

urlpatterns = [
    path('admin/', admin.site.urls),
    path('crop2/', include('crop2.urls')),  # crop2 অ্যাপের URLs
]
