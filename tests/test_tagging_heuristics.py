"""Tests for heuristic H1 tagging."""

import pytest

from mcp_code_rag.tagging.heuristics import tag_h1


class TestTagH1:
    """Test cases for tag_h1 heuristic."""

    def test_test_file_python(self):
        """Test file detection for Python test files."""
        tags = tag_h1("tests/test_auth.py", "python")
        assert "lang:python" in tags
        assert "type:test" in tags
        assert len(tags) == len(set(tags))  # No duplicates

    def test_api_router(self):
        """Test API layer detection for router files."""
        tags = tag_h1("api/router.py", "python")
        assert "lang:python" in tags
        assert "layer:api" in tags
        assert len(tags) == len(set(tags))

    def test_config_yaml(self):
        """Test config file detection for YAML files."""
        tags = tag_h1("config.yaml", "config")
        assert "lang:config" in tags
        assert "type:config" in tags
        assert len(tags) == len(set(tags))

    def test_doc_markdown(self):
        """Test documentation detection for markdown files."""
        tags = tag_h1("docs/README.md", "doc")
        assert "lang:doc" in tags
        assert "type:doc" in tags
        assert len(tags) == len(set(tags))

    def test_no_language(self):
        """Test tag_h1 with no language detected."""
        tags = tag_h1("unknown.xyz", None)
        assert "lang:" not in " ".join(tags)
        assert len(tags) == len(set(tags))

    def test_model_file(self):
        """Test model layer detection."""
        tags = tag_h1("models/user.py", "python")
        assert "lang:python" in tags
        assert "layer:model" in tags

    def test_model_by_name(self):
        """Test model detection by filename."""
        tags = tag_h1("user_model.py", "python")
        assert "layer:model" in tags

    def test_service_file(self):
        """Test service layer detection."""
        tags = tag_h1("services/auth.py", "python")
        assert "lang:python" in tags
        assert "layer:service" in tags

    def test_middleware_file(self):
        """Test middleware layer detection."""
        tags = tag_h1("middleware/auth.py", "python")
        assert "lang:python" in tags
        assert "layer:middleware" in tags

    def test_middleware_by_name(self):
        """Test middleware detection by filename."""
        tags = tag_h1("auth_middleware.py", "python")
        assert "layer:middleware" in tags

    def test_utils_file(self):
        """Test utils layer detection."""
        tags = tag_h1("utils/helpers.py", "python")
        assert "lang:python" in tags
        assert "layer:utils" in tags

    def test_test_with_underscore_test(self):
        """Test detection of files ending with _test."""
        tags = tag_h1("auth_test.py", "python")
        assert "type:test" in tags

    def test_test_with_underscore_spec(self):
        """Test detection of files ending with _spec."""
        tags = tag_h1("auth_spec.js", "javascript")
        assert "type:test" in tags

    def test_test_directory_detection(self):
        """Test detection via test/ directory."""
        tags = tag_h1("test/integration/user_test.py", "python")
        assert "type:test" in tags

    def test_config_json(self):
        """Test config detection for JSON files."""
        tags = tag_h1("config.json", "config")
        assert "type:config" in tags

    def test_config_toml(self):
        """Test config detection for TOML files."""
        tags = tag_h1("pyproject.toml", "config")
        assert "type:config" in tags

    def test_doc_folder_txt(self):
        """Test doc detection in docs folder."""
        tags = tag_h1("docs/guide.txt", "doc")
        assert "type:doc" in tags

    def test_api_routes(self):
        """Test API detection via routes directory."""
        tags = tag_h1("api/routes/user.py", "python")
        assert "layer:api" in tags

    def test_api_controllers(self):
        """Test API detection via controllers directory."""
        tags = tag_h1("api/controllers/auth.js", "javascript")
        assert "layer:api" in tags

    def test_routes_by_name(self):
        """Test API detection by routes in filename."""
        tags = tag_h1("user_routes.py", "python")
        assert "layer:api" in tags

    def test_entities_in_models(self):
        """Test model detection via entities directory."""
        tags = tag_h1("models/entities/user.py", "python")
        assert "layer:model" in tags

    def test_schemas_in_models(self):
        """Test model detection via schemas directory."""
        tags = tag_h1("models/schemas/user.ts", "typescript")
        assert "layer:model" in tags

    def test_helpers_in_utils(self):
        """Test utils detection via helpers directory."""
        tags = tag_h1("utils/helpers/string.py", "python")
        assert "layer:utils" in tags

    def test_no_duplicates_multiple_patterns(self):
        """Test that multiple matching patterns don't create duplicates."""
        tags = tag_h1("services/lib/auth.py", "python")
        assert len(tags) == len(set(tags))

    def test_sorted_output(self):
        """Test that output is sorted."""
        tags = tag_h1("tests/test_auth.py", "python")
        assert tags == sorted(tags)

    def test_env_file(self):
        """Test config detection for .env files."""
        tags = tag_h1(".env.local", "config")
        assert "type:config" in tags

    def test_xml_config(self):
        """Test config detection for XML files."""
        tags = tag_h1("config.xml", "config")
        assert "type:config" in tags

    def test_interceptor_middleware(self):
        """Test middleware detection by interceptor name."""
        tags = tag_h1("auth_interceptor.ts", "typescript")
        assert "layer:middleware" in tags

    def test_complex_path(self):
        """Test complex path with multiple layers."""
        tags = tag_h1("src/api/routes/user_model.py", "python")
        assert "lang:python" in tags
        assert "layer:api" in tags
        # user_model matches "model" in filename
        assert "layer:model" in tags

    def test_no_false_positives(self):
        """Test that similar patterns don't create false positives."""
        tags = tag_h1("modular.py", "python")
        # "modular" contains "model" but shouldn't match
        assert "layer:model" not in tags

    def test_db_folder_in_models(self):
        """Test model detection via db directory."""
        tags = tag_h1("models/db/migrations.py", "python")
        assert "layer:model" in tags

    def test_core_service(self):
        """Test service detection via core directory."""
        tags = tag_h1("core/auth.py", "python")
        assert "layer:service" in tags

    def test_business_service(self):
        """Test service detection via business directory."""
        tags = tag_h1("business/logic.py", "python")
        assert "layer:service" in tags

    def test_shared_utils(self):
        """Test utils detection via shared directory."""
        tags = tag_h1("shared/constants.py", "python")
        assert "layer:utils" in tags

    def test_views_api(self):
        """Test API detection via views directory."""
        tags = tag_h1("api/views/users.py", "python")
        assert "layer:api" in tags

    def test_tests_directory_variant(self):
        """Test detection via tests/ (plural) directory."""
        tags = tag_h1("src/tests/user.test.ts", "typescript")
        assert "type:test" in tags

    def test_double_underscore_tests(self):
        """Test detection via __tests__ directory."""
        tags = tag_h1("src/__tests__/component.test.js", "javascript")
        assert "type:test" in tags

    def test_yml_config(self):
        """Test config detection for .yml files."""
        tags = tag_h1("docker-compose.yml", "config")
        assert "type:config" in tags

    def test_cfg_config(self):
        """Test config detection for .cfg files."""
        tags = tag_h1("setup.cfg", "config")
        assert "type:config" in tags

    def test_ini_config(self):
        """Test config detection for .ini files."""
        tags = tag_h1("settings.ini", "config")
        assert "type:config" in tags

    def test_chunk_parameter_ignored(self):
        """Test that chunk parameter is accepted but ignored."""
        tags1 = tag_h1("test/file.py", "python")
        tags2 = tag_h1("test/file.py", "python", "some chunk content")
        assert tags1 == tags2
