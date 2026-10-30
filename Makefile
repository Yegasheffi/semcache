.PHONY: test serve demo report clean

test:
	python3 semcache.py selftest

serve:
	python3 semcache.py serve

demo:
	python3 semcache.py demo --db /tmp/semcache-demo.db --keep

report:
	python3 semcache.py report

clean:
	rm -rf __pycache__ .pytest_cache
