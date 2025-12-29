from mongoengine import DynamicDocument


class Cities(DynamicDocument):
    meta = {'collection': 'cities'}