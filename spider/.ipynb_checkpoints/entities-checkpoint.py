from scrapy.item import Item, Field


class TranscriptData(Item):
    title = Field()
    url = Field()
    date = Field()
    company = Field()

    executives = Field()
    analysts = Field()
    body = Field()