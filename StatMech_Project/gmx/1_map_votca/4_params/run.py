from pathlib import Path
import settings


mapping_file = Path(__file__).resolve().parent.parent / "mapping" / "mapping.xml"
settings.generate_settings(mapping_file, "settings.xml")