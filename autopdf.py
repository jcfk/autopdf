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


class ConfirmPDFSectionPagenum(BaseModel):
    confirmed: bool


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


def parse_pdf_toc_naive(fpath):
    with open(fpath, "rb") as f:
        data = f.read()

    data_b64 = base64.b64encode(data).decode("utf-8")
    file_data = {"fname": fpath.name, "data_b64": data_b64}

    message = (
        "Parse the table of contents from the content of the PDF, which may"
        " be  an academic  paper or  a  book. Detect  section titles,  their"
        " page numbers, and  their child subsections."
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


def confirm_section_pagenum(section, fpath, pagenum):
    with tempfile.TemporaryFile() as f:
        reader = PdfReader(fpath)
        writer = PdfWriter()
        writer.add_page(reader.pages[pagenum])
        writer.write(f)

        f.seek(0)
        data = f.read()

    data_b64 = base64.b64encode(data).decode("utf-8")
    file_data = {"fname": fpath.name, "data_b64": data_b64}

    message = (
        "You  are part  of  a  system that  automatically  generates tables  of"
        " contents for  books and  articles by  scanning through  individual PDF"
        " pages and  locating section  headers. You  will be  given 1)  a header"
        " pair, consisting of a section title and a page number, and 2) a single"
        " candidate page of a PDF."
        " Determine  whether or  not the  header pair  corresponds to  the given"
        " candidate page,  namely whether the  section named in the  header pair"
        " actually begins  on the given page.  This is true, for  example, if"
        " the page  contains large,  header-shaped text  identical to  the given"
        " section title and the page number shown in the page is identical to"
        " the given page number."
        " Format  your response  as a  json object  containing a  single boolean"
        " field 'confirmed', which is true iff the candidate page is a match."
        " The header pair is as follows:"
        f" Title: '{section.title}'"
        f" Page number: '{section.pagenum}'"
    )

    output = llm_helpful_assistant(
        message, file_data, response_format=ConfirmPDFSectionPagenum
    )

    # breakpoint()
    return output.confirmed


def adjust_section_pagenum(section, fpath, physical_offset):
    fpath_len = 259
    pages = list(range(fpath_len))
    pages.reverse()

    # Note section.pagenum is probably 1-indexed
    search_start_i = section.pagenum - 1 + physical_offset
    while True:
        # Search forward, then backward, with an increasing radius
        current_search_i = min(
            pages, key=lambda x: abs(x - search_start_i)
        )
        pages.remove(current_search_i)

        print(
            f"Searching for section {section.title} (pagenum: {section.pagenum}) at PDF page {current_search_i}"
        )
        if confirm_section_pagenum(section, fpath, current_search_i):
            break

    section.pagenum = current_search_i + 1
    return physical_offset + current_signed_radius


def parse_pdf_toc(fpath):
    toc = parse_pdf_toc_naive(fpath)
    print("Parsed naive toc")

    # Adjust page numbers
    def flatten_toc(sections):
        flat_toc = []
        for section in sections:
            flat_toc.append(section)
            flat_toc.extend(flatten_toc(section.subsections))

        return flat_toc

    flat_naive_toc = []
    for section in toc.sections:
        flat_naive_toc.append(section)
        flat_naive_toc.extend(flatten_toc(section.subsections))

    # breakpoint()

    current_physical_offset = 0
    for section in flat_naive_toc:
        current_physical_offset = adjust_section_pagenum(
            section, fpath, current_physical_offset
        )
        print(f"Confirmed section {section.title} (at page {section.pagenum})")
        # breakpoint()

    return toc


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
