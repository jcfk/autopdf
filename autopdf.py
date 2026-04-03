# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "openai",
#     "pydantic",
#     "pypdf",
# ]
# ///
import argparse
import base64
import tempfile
from pathlib import Path

from openai import OpenAI
from pypdf import PdfReader, PdfWriter
from pydantic import BaseModel


class PDFMetadata(BaseModel):
    year: int
    title: str
    author_surnames: list[str]
    error: bool


def llm_helpful_assistant(prompt, file_data=None, response_format=None):
    client = OpenAI()

    # I have no idea  what this api is. It's not in the  api reference, but it's
    # in the structured outputs page: https://developers.openai.com/api/docs/guides/structured-outputs
    response = client.responses.parse(
        model="gpt-5.4-nano",
        text_format=response_format,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {
                        "type": "input_file",
                        "filename": file_data["fname"],
                        # https://community.openai.com/t/issue-with-native-pdf-input-base64-in-gpt-4o-error-invalid-input-0-content-1-file-data/1377908
                        "file_data": f"data:application/pdf;base64,{file_data['data_b64']}",
                    },
                ],
            }
        ],
    )

    # ???
    return response.output_parsed


def parse_pdf_metadata(fpath, last_page):
    # with temp pdf
    with tempfile.TemporaryFile() as f:
        reader = PdfReader(fpath)
        writer = PdfWriter()
        for i in range(min(len(reader.pages), last_page)):
            writer.add_page(reader.pages[i])
        writer.write(f)

        f.seek(0)
        data = f.read()
        data_b64 = base64.b64encode(data).decode("utf-8")
        file_data = {"fname": fpath.name, "data_b64": data_b64}

        message = (
            "Detect the metadata  for year, title, and author  surnames from the"
            " following text of the first pages of an academic paper or book."
            " Format your  response as a  json object,  where 'year' is  an int,"
            " 'title' is a  string, 'authors' is a list of  surname strings, and"
            " 'error' is a boolean  that is true if and only  if the task cannot"
            " be completed. Return error if the document does not possess all"
            " three fields."
            " The title  should be normalized  if necessary. That means,  if the"
            " title is in all caps in the pdf, capitalization should be adjusted"
            " to   standard   titlecase.   Capitalized   acronyms   may   remain"
            " capitalized. Retain necessary formatting  elements like colons and"
            " commas."
            " The author list should include surnames in the order in which they"
            " appear  in the  document.  The first  author's  surname should  be"
            " first. The same normalization rules apply to author surnames."
        )

        # breakpoint()
        output = llm_helpful_assistant(message, file_data, response_format=PDFMetadata)

        return output


def rename_pdf(fpath, metadata):
    year = metadata.year
    title = metadata.title
    first_author = metadata.author_surnames[0]
    multi_author = len(metadata.author_surnames) > 1

    new_fname = f"({year}) {title} ({first_author}{" et al." if multi_author else ""}).pdf"
    new_fpath = Path(fpath.parent, new_fname)

    if not new_fpath.exists():
        fpath.rename(new_fpath)
        print(f"Renamed {fpath} --> {new_fpath}")

def main():
    parser = argparse.ArgumentParser("autopdf")
    parser.add_argument("fpath", nargs="+")
    parser.add_argument(
        "--last-page",
        type=int,
        default="4",
        help="extract metadata from up to this page, 0-indexed",
    )
    args = parser.parse_args()

    for fpath in args.fpath:
        fpath = Path(fpath)
        metadata = parse_pdf_metadata(fpath, last_page=args.last_page)
        if metadata.error:
            print(f"Error processing {fpath}; skipping")
            continue

        new_fpath = rename_pdf(fpath, metadata)


if __name__ == "__main__":
    main()
