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
import sys
from pathlib import Path

from openai import OpenAI
from pypdf import PdfReader, PdfWriter
from pydantic import BaseModel


class ParsedPDFMetadata(BaseModel):
    year: int
    title: str
    author_surnames: list[str]
    error: bool


class ParsedPDFTocSection(BaseModel):
    title: str
    pagenum: int
    subsections: list["ParsedPDFTocSection"]


class ParsedPDFTocTopLevel(BaseModel):
    sections: list[ParsedPDFTocSection]


def err(msg=None):
    sys.exit(msg if msg else 1)


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
            f" You may take into account the file name, which is '{fpath.name}'."
        )

        # breakpoint()
        output = llm_helpful_assistant(
            message, file_data, response_format=ParsedPDFMetadata
        )

        return output


def rename_pdf(fpath, metadata):
    year = metadata.year
    title = metadata.title
    first_author = metadata.author_surnames[0]
    multi_author = len(metadata.author_surnames) > 1

    new_fname = (
        f"({year}) {title} ({first_author}{' et al.' if multi_author else ''}).pdf"
    )
    new_fpath = Path(fpath.parent, new_fname)

    if not new_fpath.exists():
        fpath.rename(new_fpath)
        print(f"Renamed {fpath}")
        print(f"    --> {new_fpath}")
        print()


def check_pdf_toc_exists(fpath):
    reader = PdfReader(fpath)
    return True if len(reader.outline) > 0 else False


def parse_pdf_toc(fpath):
    with open(fpath, "rb") as f:
        data = f.read()

    data_b64 = base64.b64encode(data).decode("utf-8")
    file_data = {"fname": fpath.name, "data_b64": data_b64}

    message = (
        "Parse the table of contents from the content of the PDF, which may"
        " be  an academic  paper or  a  book. Detect  section titles,  their"
        " physical page numbers, and  their child subsections. Do not"
        " merely parse  the in-text  table of  contents; ensure  parsed page"
        " numbers correspond to page numbers in the physical PDF, 1-indexed."
        " Include administrative  sections such as the  preface, references,"
        " and index, if any exist."
        " Format  your response  as a  nested top  level json  object, where"
        " 'sections'  is a  list  of top-level  toc  components. Each  toc"
        " component is  a json object  where 'title' is a  string containing"
        " the section  title, 'pagenum'  is an  int containing  the physical"
        " page, and 'subsections'  is a list of child toc  components, in the"
        " order in which they appear."
        " For example,  subchapters of  a book should  be recorded  as child"
        " sections of top-level  chapters,  which should  be recorded  as"
        " child sections of major parts, if any exist."
        " Normalize   section  titles   into   titlecase,  with   reasonable"
        " approximations of mathematical notation by ASCII characters."
    )

    output = llm_helpful_assistant(
        message, file_data, response_format=ParsedPDFTocTopLevel
    )

    return output


def pprint_toc(toc):
    def pprint_toc_sub(section, level):
        print(" " * (level - 1) + f"- {section.title} (page: {section.pagenum})")
        for subsection in section.subsections:
            pprint_toc_sub(subsection, level + 1)

    for section in toc.sections:
        pprint_toc_sub(section, 1)


def apply_pdf_toc(fpath, toc):
    # Delete existing toc
    # See: https://github.com/py-pdf/pypdf/discussions/1427
    reader = PdfReader(fpath)
    writer = PdfWriter()
    writer.append_pages_from_reader(reader)

    # Apply new toc
    def apply_pdf_toc_sub(writer, sections, parent=None):
        for section in sections:
            section_outline_item = writer.add_outline_item(
                title=section.title, page_number=section.pagenum, parent=parent
            )
            apply_pdf_toc_sub(writer, section.subsections, parent=section_outline_item)

    apply_pdf_toc_sub(writer, toc.sections)
    writer.write(fpath)


def main():
    parser = argparse.ArgumentParser("autopdf")
    parser.add_argument("cmd")
    parser.add_argument("fpath", nargs="+")
    parser.add_argument(
        "--last-page",
        type=int,
        default=4,
        help="extract metadata from up to this page, 0-indexed",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    for fpath in args.fpath:
        fpath = Path(fpath)

        if args.cmd == "rename":
            parsed_metadata = parse_pdf_metadata(fpath, last_page=args.last_page)
            if parsed_metadata.error:
                print(f"Error processing {fpath}; skipping")
                print()
                continue

            new_fpath = rename_pdf(fpath, metadata)
        elif args.cmd == "make-toc":
            if check_pdf_toc_exists(fpath) and not args.force:
                print(f"PDF has TOC (use --force): {fpath}; skipping")
                print()
                continue

            toc = parse_pdf_toc(fpath)
            apply_pdf_toc(fpath, toc)
        else:
            err(f"Unknown cmd {args.cmd}")


if __name__ == "__main__":
    main()
