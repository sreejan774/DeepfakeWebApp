from django.core.exceptions import ValidationError


def FileSizeValidator(value):
    filesize = value.size
    if filesize > 10000000:
        raise ValidationError("Maximum upload size is 10 MB")
