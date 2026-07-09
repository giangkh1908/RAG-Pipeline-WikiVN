import unittest

from rag_pipeline.transform.cleaner import WikipediaArticleCleaner


class CleanerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cleaner = WikipediaArticleCleaner()

    # ── Existing v1 tests (must still pass) ─────────────────────────

    def test_removes_standard_markup(self) -> None:
        text = (
            "__NOTOC__\n"
            "{{Infobox}}\n"
            "Tiếng Việt là ngôn ngữ.\n"
            "<ref>tham khảo</ref>\n"
            "[[Thể loại:Ngôn ngữ]]\n"
            "Được sử dụng rộng rãi.\n"
        )

        cleaned = self.cleaner.clean(text)

        self.assertNotIn("__NOTOC__", cleaned)
        self.assertNotIn("{{Infobox}}", cleaned)
        self.assertNotIn("<ref>tham khảo</ref>", cleaned)
        self.assertIn("Tiếng Việt là ngôn ngữ.", cleaned)
        self.assertIn("Được sử dụng rộng rãi.", cleaned)

    def test_removes_pipe_lines_and_closing_braces(self) -> None:
        text = (
            "| purpose = Đảm bảo phát triển Internet.\n"
            "| region_served = Toàn cầu\n"
            "}}\n"
            "Internet Society là tổ chức quốc tế.\n"
        )

        cleaned = self.cleaner.clean(text)

        self.assertNotIn("| purpose", cleaned)
        self.assertNotIn("| region_served", cleaned)
        self.assertNotIn("}}", cleaned)
        self.assertIn("Internet Society là tổ chức quốc tế.", cleaned)

    def test_unwraps_wikilinks(self) -> None:
        text = "Xem thêm [[Việt Nam|Việt Nam]] và [[Lịch sử Việt Nam]]."

        cleaned = self.cleaner.clean(text)

        self.assertNotIn("[[", cleaned)
        self.assertNotIn("]]", cleaned)
        self.assertIn("Việt Nam", cleaned)
        self.assertIn("Lịch sử Việt Nam", cleaned)

    def test_strips_bold_italic(self) -> None:
        text = "Đây là '''chữ đậm''' và ''chữ nghiêng''."

        cleaned = self.cleaner.clean(text)

        self.assertNotIn("'''", cleaned)
        self.assertNotIn("''", cleaned)
        self.assertIn("chữ đậm", cleaned)
        self.assertIn("chữ nghiêng", cleaned)

    def test_handles_infobox_before_prose(self) -> None:
        """Real-world case: entry starts with template artifacts then prose."""
        text = (
            "(Hà Nội)\n"
            "(Huế)\n"
            "~ (TP. Hồ Chí Minh)\n"
            "}}\n"
            "| speakers = L1: triệu\n"
            "| ethnicity = Việt (Kinh)\n"
            "\n"
            "Tiếng Việt hay tiếng Kinh là một ngôn ngữ.\n"
        )

        cleaned = self.cleaner.clean(text)

        self.assertNotIn("| speakers", cleaned)
        self.assertNotIn("| ethnicity", cleaned)
        self.assertNotIn("}}", cleaned)
        self.assertIn("Tiếng Việt hay tiếng Kinh là một ngôn ngữ.", cleaned)

    def test_entry_with_only_template_garbage_has_no_markup(self) -> None:
        """Trang_Chính-style: only infobox links, no real prose — markup removed."""
        text = (
            "__NOTOC____NOEDITSECTION__\n"
            "| links =\n"
            "Lưu trữ\n"
            "Thêm bài viết chọn lọc\n"
            "Ứng cử viên\n"
            "}}\n"
        )

        cleaned = self.cleaner.clean(text)

        self.assertNotIn("| links", cleaned)
        self.assertNotIn("__NOTOC__", cleaned)
        self.assertNotIn("}}", cleaned)

    def test_already_clean_text_is_unchanged(self) -> None:
        text = "World Wide Web Consortium (W3C) là tổ chức tiêu chuẩn quốc tế."

        cleaned = self.cleaner.clean(text)

        self.assertEqual(text, cleaned)

    # ── v2 new tests ────────────────────────────────────────────────

    def test_removes_nested_templates(self) -> None:
        """{{Infobox | image = {{Image | url=x}} }} → all removed."""
        text = (
            "{{Infobox\n"
            "| image = {{Hình ảnh\n"
            "| url = test.jpg\n"
            "| caption = Mô tả\n"
            "}}\n"
            "}}\n"
            "Đây là nội dung thật.\n"
        )

        cleaned = self.cleaner.clean(text)

        self.assertNotIn("Infobox", cleaned)
        self.assertNotIn("Hình ảnh", cleaned)
        self.assertNotIn("url", cleaned.lower())
        self.assertNotIn("caption", cleaned.lower())
        self.assertNotIn("test.jpg", cleaned)
        self.assertIn("Đây là nội dung thật.", cleaned)

    def test_removes_double_nested_templates(self) -> None:
        """Three-level nesting: {{{1|{{2|{{3}}}}}} → all removed."""
        text = (
            "{{outer\n"
            "| field = {{middle\n"
            "| sub = {{inner|value}}\n"
            "}}\n"
            "}}\n"
            "Nội dung sau template.\n"
        )

        cleaned = self.cleaner.clean(text)

        self.assertNotIn("outer", cleaned)
        self.assertNotIn("middle", cleaned)
        self.assertNotIn("inner", cleaned)
        self.assertNotIn("value", cleaned)
        self.assertIn("Nội dung sau template.", cleaned)

    def test_removes_multiline_infobox_value_continuations(self) -> None:
        """| links =\nLưu trữ\nThêm bài → all removed, not just first line."""
        text = (
            "{{Đầu trang\n"
            "| links =\n"
            "Lưu trữ\n"
            "Thêm bài viết chọn lọc\n"
            "Ứng cử viên\n"
            "}}\n"
        )

        cleaned = self.cleaner.clean(text)

        self.assertNotIn("Lưu trữ", cleaned)
        self.assertNotIn("Thêm bài viết chọn lọc", cleaned)
        self.assertNotIn("Ứng cử viên", cleaned)

    def test_does_not_remove_real_content_after_infobox(self) -> None:
        """Real content after infobox section should survive."""
        text = (
            "{{Infobox\n"
            "| tên = Hà Nội\n"
            "| diện tích = 3359 km²\n"
            "}}\n"
            "\n"
            "Hà Nội là thủ đô của nước Cộng hòa xã hội chủ nghĩa Việt Nam.\n"
            "Đây là thành phố trực thuộc trung ương.\n"
        )

        cleaned = self.cleaner.clean(text)

        self.assertIn("Hà Nội là thủ đô", cleaned)
        self.assertIn("thành phố trực thuộc trung ương", cleaned)

    def test_keeps_list_items_after_infobox(self) -> None:
        """List items (*, -, #) after infobox should not be treated as noise."""
        text = (
            "{{Infobox\n"
            "| tên = Tỉnh\n"
            "| diện tích = 5000\n"
            "}}\n"
            "* Điểm cực bắc: xã A\n"
            "* Điểm cực nam: xã B\n"
            "Vị trí địa lý của tỉnh này rất đa dạng.\n"
        )

        cleaned = self.cleaner.clean(text)

        self.assertIn("Điểm cực bắc", cleaned)
        self.assertIn("Điểm cực nam", cleaned)
        self.assertIn("Vị trí địa lý", cleaned)

    def test_handles_template_with_equals_in_value(self) -> None:
        """Template with = in value should not break removal."""
        text = (
            "{{Thông tin\n"
            "| công thức = E = mc²\n"
            "| mô tả = Năng lượng = khối lượng × vận tốc²\n"
            "}}\n"
            "Công thức này do Einstein đề xuất.\n"
        )

        cleaned = self.cleaner.clean(text)

        self.assertNotIn("công thức", cleaned)
        self.assertNotIn("E = mc²", cleaned)
        self.assertIn("Công thức này do Einstein đề xuất.", cleaned)


if __name__ == "__main__":
    unittest.main()
