from scrapy.exporters import CsvItemExporter, JsonItemExporter

class DataPipeline(object):
    def process_item(self, item, spider):
        return item


class JsonWriterPipeline(object):

    def __init__(self,spider):
        self.file = open('../Resources/transcript.json', 'wb')
        self.exporter = JsonItemExporter(self.file, encoding="utf-8", ensure_ascii=False)
        self.exporter.start_exporting()

    def open_spider(self, spider):
        pass

    def close_spider(self, spider):
        self.exporter.finish_exporting()
        self.file.close()
        

    def process_item(self, item, spider):
        self.exporter.export_item(item)
        return item