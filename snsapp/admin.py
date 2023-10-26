from django.contrib import admin
from .models import Post,Connection, Comment

# Register your models here.
admin.site.register(Post)
admin.site.register(Connection)
admin.site.register(Comment)