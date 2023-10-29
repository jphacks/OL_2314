from django.db import models
from django.contrib.auth.models import User

# Create your models here.
class Post(models.Model):
    content = models.TextField()
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    like = models.ManyToManyField(User, related_name='related_post', blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    face_emotion = models.CharField(max_length=20, blank=True, null=True)
    text_emotion = models.CharField(max_length=20, blank=True, null=True)

    def __str__(self): #データベースで表示されるもの
        return self.content

    @property
    def comment_count(self):
        return Comment.objects.filter(post=self).count()

    class Meta:
        ordering = ["-created_at"]


class Connection(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    following = models.ManyToManyField(User, related_name='following', blank=True)

    def __str__(self):
        return self.user.username

class Comment(models.Model): #コメント機能
    post = models.ForeignKey(Post, on_delete=models.CASCADE, related_name='comments') #Postモデルに対して、1対多の関係を定義
    user = models.ForeignKey(User, on_delete=models.CASCADE) #Userモデルに対して、1対多の関係を定義
    content = models.TextField() #コメントの内容
    created_at = models.DateTimeField(auto_now_add=True) #コメントの作成日時
    face_emotion = models.CharField(max_length=20, blank=True, null=True)
    text_emotion = models.CharField(max_length=20, blank=True, null=True)

    def __str__(self): #データベースで表示されるもの
        return self.content

    class Meta: #データベースで表示されるもの
        ordering = ["-created_at"]