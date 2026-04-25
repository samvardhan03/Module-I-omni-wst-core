.PHONY: build test memcheck clean

build:
	pip install -e .

test:
	pytest tests/ -v

memcheck:
	@echo "Running compute-sanitizer for memory safety validation..."
	compute-sanitizer --tool memcheck pytest tests/ -v

clean:
	rm -rf build/
	rm -rf _skbuild/
	rm -rf *.egg-info
