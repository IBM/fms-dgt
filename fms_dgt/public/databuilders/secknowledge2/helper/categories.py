# Standard
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import json


@dataclass
class SubCategory:
    """Class representing a data subcategory."""

    name: str
    """Used to determine the subcategory when fetching the row or for classification prompt."""

    description: str
    """Used to determine the subcategory in the classification prompt."""

    structure: str = (
        """First, analyse the question and give a brief analysis in the first paragraph. Then output the answer. Next, use a list to give explanations. Last, give a conclusion."""
    )
    """Detailed instructions on how to rewrite the answer."""

    requires_search: bool = False
    requires_grounding_doc: bool = False
    requires_rewrite: bool = False


@dataclass
class Category:
    """Class representing a data category."""

    name: str
    subcategories: list[SubCategory]

    def __str__(self):
        """Return the string representation of the category."""

        return (
            self.name
            + " (examples: "
            + ", ".join([subcategory.name for subcategory in self.subcategories])
            + ")"
        )

    def subcategories_str(self):
        """Return the string representation of the subcategories."""

        output = ""
        for i, subcategory in enumerate(self.subcategories):
            output += f"{i+1}. {subcategory.name}: {subcategory.description}\n"
        return output


class TemplateData:
    def __init__(self, categories: List[Category]):
        self.categories = categories

    @staticmethod
    def from_auto(path: Path) -> "TemplateData":
        if path.is_file():
            return TemplateData.from_json(path)
        elif path.is_dir():
            return TemplateData.from_dir(path)
        else:
            raise ValueError(f"Invalid templates path: {path}.")

    @staticmethod
    def from_json(json_path: Path) -> "TemplateData":
        """Load the categories from a JSON file."""

        with open(json_path, "r") as f:
            data = json.load(f)

        categories = []
        for category_data in data:
            subcategories = [
                SubCategory(**subcategory_data)
                for subcategory_data in category_data["subcategories"]
            ]
            categories.append(Category(category_data["name"], subcategories))

        return TemplateData(categories)

    @staticmethod
    def from_dir(templates_dir: Path) -> "TemplateData":
        """Load the categories from a directory of JSON files."""

        categories = []
        for json_path in templates_dir.glob("*.json"):
            categories.extend(TemplateData.from_json(json_path).categories)

        return TemplateData(categories)

    def categories_str(self):
        """Return the string representation of the categories."""

        output = ""
        for i, category in enumerate(self.categories):
            output += f"{i+1}. {str(category)}\n"
        return output

    def categories_names(self) -> List[str]:
        """Return the list of category names."""

        return [category.name for category in self.categories]

    def subcategories_names(self) -> List[str]:
        """Return the list of subcategories names."""

        return [
            subcategory.name
            for category in self.categories
            for subcategory in category.subcategories
        ]

    def get_category_by_name(self, name: str) -> Category:
        """Return the category object by its name."""

        for category in self.categories:
            if category.name == name:
                return category
        raise ValueError(f"Category '{name}' not found.")

    def get_subcategory_by_name(
        self, name: str, category: Optional[Category] = None
    ) -> SubCategory:
        """Return the subcategory object by its name."""

        if category:
            for subcategory in category.subcategories:
                if subcategory.name == name:
                    return subcategory
            raise ValueError(f"Subcategory '{name}' not found in category '{category.name}'.")
        else:
            for category in self.categories:
                for subcategory in category.subcategories:
                    if subcategory.name == name:
                        return subcategory
            raise ValueError(f"Subcategory '{name}' not found.")
