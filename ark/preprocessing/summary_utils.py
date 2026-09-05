import json
import re
import numpy as np
import pandas as pd

MAX_ENTRIES = 25


def add_summary_amazon(row) -> str:
    if row["type"] == "brand":
        return f"brand name: {row['brand_name']}"
    if row["type"] == "category":
        return f"category name: {row['category_name']}"
    if row["type"] == "color":
        return f"color name: {row['color_name']}"

    summary = f"- product: {row['title']}\n"
    if row["brand"]:
        summary += f"- brand: {row['brand']}\n"
    if row["details"] and str(row["details"]) != "nan":
        if type(row["details"]) == str:
            details = eval(row["details"])  # TODO: I think this line never runs.
        else:
            details = row["details"]
        if details.get("product_dimensions"):
            try:
                dimensions, weight = details["product_dimensions"].split(" ; ")
                summary += f"- dimensions: {dimensions}\n- weight: {weight}\n"
            except ValueError:
                pass

    if row["description"]:
        if type(row["description"]) == str:
            row_description = eval(row["description"])  # TODO: I think this line never runs.
        else:
            row_description = row["description"]
        description = " ".join(row_description).strip(" ")
        if description:
            summary += f"- description: {description}\n"

    feature_text = "- features: \n"
    if row["feature"]:
        if type(row["feature"]) == str:
            row_feature = eval(row["feature"])
        else:
            row_feature = row["feature"]
        for feature_idx, feature in enumerate(row_feature):
            if feature and "asin" not in feature.lower():
                feature_text += f"#{feature_idx + 1}: {feature}\n"
    else:
        feature_text = ""

    if row["review"]:
        if type(row["review"]) == str:
            row_review = json.loads(row["review"])
        else:
            row_review = row["review"]  # TODO: I think this line never runs.
        review_text = "- reviews: \n"
        scores = [
            0 if pd.isnull(review["vote"]) else int(review["vote"].replace(",", ""))
            for review in row_review
        ]
        ranks = np.argsort(-np.array(scores))
        for i, review_idx in enumerate(ranks):
            review = row_review[review_idx]
            review_text += f'#{review_idx + 1}:\nsummary: {review["summary"]}\ntext: "{review["reviewText"]}"\n'
            if i > MAX_ENTRIES:
                break
    else:
        review_text = ""

    if type(row["qa"]) == str:
        row_qa = json.loads(row["qa"])
    else:
        row_qa = row["qa"]  # TODO: I think this line never runs.
    if row_qa:
        qa_text = "- Q&A: \n"
        for qa_idx, qa in enumerate(row_qa):
            qa_text += f'#{qa_idx + 1}:\nquestion: "{qa["question"]}"\nanswer: "{qa["answer"]}"\n'
            if qa_idx > MAX_ENTRIES:
                break
    else:
        qa_text = ""

    summary += feature_text + review_text + qa_text

    # if add_rel:
    #     summary += self.get_rel_info(idx)
    # if compact:
    #     summary = compact_text(summary)

    return summary


def add_summary_to_prime(row) -> str:
    summary = f"- name: {row['name']}\n"
    summary += f"- type: {row['type']}\n"
    summary += f"- source: {row['source']}\n"

    gene_protein_text_explain = {
        "name": "gene name",
        "type_of_gene": "gene types",
        "alias": "other gene names",
        "other_names": "extended other gene names",
        "genomic_pos": "genomic position",
        "generif": "PubMed text",
        "interpro": "protein family and classification information",
        "summary": "protein summary text",
    }

    feature_text = f"- details:\n"
    feature_cnt = 0

    if row["details"]:
        for key, value in json.loads(row["details"]).items():
            if str(value) in ["", "nan"] or key.startswith("_") or "_id" in key:
                continue
            if row["type"] == "gene/protein" and key in gene_protein_text_explain.keys():
                if "interpro" in key:
                    if isinstance(value, dict):
                        value = [value]
                    value = [v["desc"] for v in value]
                if "generif" in key:
                    value = "; ".join([v["text"] for v in value])
                    value = " ".join(value.split(" ")[:50000])
                if "genomic_pos" in key:
                    if isinstance(value, list):
                        value = value[0]
                feature_text += f"  - {key} ({gene_protein_text_explain[key]}): {value}\n"
                feature_cnt += 1
            else:
                feature_text += f"  - {key}: {value}\n"
                feature_cnt += 1
    if feature_cnt == 0:
        feature_text = ""

    summary += feature_text

    # if add_rel:
    #     summary += self.get_rel_info(idx, n_rel=n_rel)
    # if compact:
    #     summary = compact_text(summary)

    return summary


def add_summary_to_mag(row) -> str:
    if row["type"] == "author":
        summary = f"- author name: {row['DisplayName']}\n"
        if row["PaperCount"] != -1:
            summary += f"- author paper count: {row['PaperCount']}\n"
        if row["CitationCount"] != -1:
            summary += f"- author citation count: {row['CitationCount']}\n"
        summary = summary.replace("-1", "Unknown")

    elif row["type"] == "paper":
        summary = f" - paper title: {row['title']}\n"
        summary += " - abstract: " + row["abstract"].replace("\r", "").rstrip("\n") + "\n"
        if str(row["Date"]) != "-1":
            summary += f" - publication date: {row['Date']}\n"
        if str(row["OriginalVenue"]) != "-1":
            summary += f" - venue: {row['OriginalVenue']}\n"
        elif str(row["JournalDisplayName"]) != "-1":
            summary += f" - journal: {row['JournalDisplayName']}\n"
        elif str(row["ConferenceSeriesDisplayName"]) != "-1":
            summary += f" - conference: {row['ConferenceSeriesDisplayName']}\n"
        elif str(row["ConferenceInstancesDisplayName"]) != "-1":
            summary += f" - conference: {row['ConferenceInstancesDisplayName']}\n"

    elif row["type"] == "field_of_study":
        summary = f" - field of study: {row['DisplayName']}\n"
        if row["PaperCount"] != -1:
            summary += f"- field paper count: {row['PaperCount']}\n"
        if row["CitationCount"] != -1:
            summary += f"- field citation count: {row['CitationCount']}\n"
        summary = summary.replace("-1", "Unknown")

    elif row["type"] == "institution":
        summary = f" - institution: {row['DisplayName']}\n"
        if row["PaperCount"] != -1:
            summary += f"- institution paper count: {row['PaperCount']}\n"
        if row["CitationCount"] != -1:
            summary += f"- institution citation count: {row['CitationCount']}\n"
        summary = summary.replace("-1", "Unknown")

    # if add_rel and row["type"] == "paper":
    #     summary += self.get_rel_info(idx, n_rel=n_rel)

    # if compact:
    #     summary = compact_text(summary)

    return summary
