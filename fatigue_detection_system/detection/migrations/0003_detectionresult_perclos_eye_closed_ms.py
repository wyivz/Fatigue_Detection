# Generated manually for perclos / eye_closed_ms fields

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("detection", "0002_detectionsession_status"),
    ]

    operations = [
        migrations.AddField(
            model_name="detectionresult",
            name="eye_closed_ms",
            field=models.IntegerField(
                blank=True, null=True, verbose_name="当前闭眼时长毫秒"
            ),
        ),
        migrations.AddField(
            model_name="detectionresult",
            name="perclos",
            field=models.FloatField(
                blank=True, null=True, verbose_name="PERCLOS百分比"
            ),
        ),
    ]
