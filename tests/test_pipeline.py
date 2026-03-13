import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import normalize_slang, is_greeting, check_order_status

class TestNormalizeSlang:
    def test_basic_slang(self):
        assert normalize_slang("gue mau beli") == "saya ingin beli"

    def test_dp_replacement(self):
        assert normalize_slang("udah dp tapi ga jadi") == "sudah down payment tapi tidak jadi"

    def test_no_slang(self):
        assert normalize_slang("saya ingin membeli produk") == "saya ingin membeli produk"

    def test_empty_string(self):
        assert normalize_slang("") == ""

    def test_filler_words_removed(self):
        result = normalize_slang("gan itu gimana")
        assert "gan" not in result
        assert "bagaimana" in result


class TestIsGreeting:
    def test_single_greeting(self):
        assert is_greeting("halo") == True

    def test_greeting_with_punctuation(self):
        assert is_greeting("halo!") == True

    def test_greeting_with_one_word(self):
        assert is_greeting("halo kak") == True

    def test_long_query_not_greeting(self):
        assert is_greeting("halo saya ingin cek order") == False

    def test_non_greeting(self):
        assert is_greeting("berapa lama pengiriman") == False

    def test_empty_string(self):
        assert is_greeting("") == False


class TestCheckOrderStatus:
    def test_valid_order(self):
        result = check_order_status("ORD001")
        assert result["success"] == True
        assert result["order_id"] == "ORD001"
        assert result["status"] == "shipped"

    def test_invalid_order(self):
        result = check_order_status("ORD999")
        assert result["success"] == False
        assert "tidak ditemukan" in result["message"]

    def test_case_insensitive(self):
        result = check_order_status("ord001")
        assert result["success"] == True

    def test_all_fields_present(self):
        result = check_order_status("ORD001")
        assert all(k in result for k in ["success", "order_id", "status", "item", "seller", "date"])