from src.utils.config import load_config


def test_default_config_loads():
    cfg = load_config("configs/default.yaml")
    assert cfg.risk.min_indicators == 2
    assert cfg.input.resolution == [640, 640]


def test_dev_config_extends_default():
    cfg = load_config("configs/dev.yaml")
    # default.yaml에서 상속
    assert cfg.risk.min_indicators == 2
    # dev.yaml에서 override
    assert cfg.logging.level == "DEBUG"


def test_pi5_config_extends_default():
    cfg = load_config("configs/pi5.yaml")
    assert cfg.input.resolution == [320, 320]
    assert cfg.input.source == "camera"
    assert cfg.detection.interval == 3
    # override되지 않은 값은 default.yaml에서 상속
    assert cfg.risk.min_indicators == 2
