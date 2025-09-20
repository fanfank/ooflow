# Release Guide for OoFlow

## Prerequisites

1. Ensure you have PyPI account and API token
2. Set up GitHub repository secrets:
   - `PYPI_API_TOKEN`: Your PyPI API token

## Release Process

### 1. Prepare Release

1. Update version in `ooflow/__init__.py`
2. Update version in `pyproject.toml` 
3. Update CHANGELOG.md (if exists)
4. Run all tests locally:
   ```bash
   python tests/run_tests.py
   ```

### 2. Create Git Tag

```bash
# Create and push tag
git tag v0.1.0
git push origin v0.1.0
```

### 3. Create GitHub Release

1. Go to GitHub repository
2. Click "Releases" → "Create a new release"
3. Choose the tag `v0.1.0`
4. Add release notes
5. Publish release

### 4. Automatic Publishing

The GitHub Action will automatically:
- Run tests on all supported Python versions
- Build the package
- Upload to PyPI

### 5. Manual Publishing (if needed)

```bash
# Install build tools
pip install build twine

# Build package
python -m build

# Check package
twine check dist/*

# Upload to PyPI
twine upload dist/*
```

## Verification

After release, verify:

```bash
# Install from PyPI
pip install ooflow==0.1.0

# Test basic functionality
python -c "import ooflow; print(ooflow.__version__)"
```

## Post-Release

1. Update version to next development version
2. Create development branch if needed
3. Update documentation if needed

## Version Numbering

Follow semantic versioning (semver):
- `MAJOR.MINOR.PATCH`
- `0.1.0` - Initial release
- `0.1.1` - Bug fixes
- `0.2.0` - New features
- `1.0.0` - Stable API
