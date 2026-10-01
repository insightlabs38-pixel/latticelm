PYTHON ?= ../.venv/bin/python
.PHONY: install preview render screenshots vhs compose qa test
install:
	./scripts/install_tools.sh
preview:
	$(PYTHON) scripts/render_all.py --quality preview
render:
	$(PYTHON) scripts/render_all.py --quality final
screenshots:
	$(PYTHON) scripts/generate_screenshots.py
vhs:
	$(PYTHON) scripts/render_vhs.py --tape verification
	$(PYTHON) scripts/render_vhs.py --tape evaluation
compose:
	$(PYTHON) scripts/compose_video.py --quality final
qa:
	$(PYTHON) scripts/validate_layout.py
	$(PYTHON) scripts/contact_sheets.py --quality preview
	$(PYTHON) -m pytest tests -q
test:
	$(PYTHON) -m pytest tests -q
