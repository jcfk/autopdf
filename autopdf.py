# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "openai",
#     "pydantic",
#     "pypdf",
#     "pdf2image"
# ]
# ///
import argparse
import base64
import tempfile
import sys
from pathlib import Path
from io import BytesIO

import pdf2image
from openai import OpenAI
from pypdf import PdfReader, PdfWriter
from pydantic import BaseModel


fpath = None


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


def llm_helpful_assistant(
    prompt, reasoning="none", media_type=None, media_data=None, response_format=None
):
    client = OpenAI()

    message_content = [
        {"type": "input_text", "text": prompt},
    ]
    if media_type == "file":
        message_content.append(
            {
                "type": "input_file",
                "filename": media_data["fname"],
                # https://community.openai.com/t/issue-with-native-pdf-input-base64-in-gpt-4o-error-invalid-input-0-content-1-file-data/1377908
                "file_data": f"data:application/pdf;base64,{media_data['data_b64']}",
            }
        )
    # https://developers.openai.com/api/docs/guides/images-vision?format=base64-encoded#analyze-images
    elif media_type == "image":
        message_content.append(
            {
                "type": "input_image",
                "image_url": f"data:image/jpeg;base64,{media_data['data_b64']}",
            }
        )

    # I have no idea  what this api is. It's not in the  api reference, but it's
    # in the structured outputs page: https://developers.openai.com/api/docs/guides/structured-outputs
    response = client.responses.parse(
        model="gpt-5.4-nano",
        reasoning={"effort": reasoning},
        text_format=response_format,
        input=[
            {
                "role": "user",
                "content": message_content,
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

        output = llm_helpful_assistant(
            message,
            media_type="file",
            media_data=file_data,
            response_format=ParsedPDFMetadata,
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


def parse_pdf_toc_naive(fpath):
    with open(fpath, "rb") as f:
        data = f.read()

    data_b64 = base64.b64encode(data).decode("utf-8")
    file_data = {"fname": fpath.name, "data_b64": data_b64}

    message = (
        "Parse the table of contents of the PDF, which may be an academic paper"
        " or a book. Detect section titles,  their page numbers, and their child"
        " subsections."
        " Include administrative  sections such as the  preface, references, and"
        " index, if any exist."
        " Format  your  response  as  a  nested top  level  json  object,  where"
        " 'sections' is a  list of top-level toc components.  Each toc component"
        " is a  json object  where 'title'  is a  string containing  the section"
        " title,  'pagenum'  is  an  int   containing  the  physical  page,  and"
        " 'subsections' is a list of child toc components, in the order in which"
        " they appear."
        " For  example,  subchapters of  a  book  should  be recorded  as  child"
        " sections  of top-level  chapters, which  should be  recorded as  child"
        " sections of major parts, if any exist."
        " Normalize   section    titles   into   titlecase,    with   reasonable"
        " approximations of mathematical notation by ASCII characters."
    )

    output = llm_helpful_assistant(
        message,
        media_type="file",
        media_data=file_data,
        response_format=ParsedPDFTocTopLevel,
    )

    return output


def confirm_section_pagenum(section, page_data, data_type):
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
    if data_type == "file":
        with tempfile.TemporaryFile() as f:
            writer = PdfWriter()
            writer.add_page(page_data)
            writer.write(f)

            f.seek(0)
            data = f.read()

        data_b64 = base64.b64encode(data).decode("utf-8")
        file_data = {"fname": fpath.name, "data_b64": data_b64}

        output = llm_helpful_assistant(
            message,
            reasoning="medium",
            media_type="file",
            media_data=file_data,
            response_format=ConfirmPDFSectionPagenum,
        )
    elif data_type == "img":
        buf = BytesIO()
        page_data.save(buf, format="JPEG")
        data_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        img_data = {"data_b64": data_b64}

        output = llm_helpful_assistant(
            message,
            reasoning="medium",
            media_type="image",
            media_data=img_data,
            response_format=ConfirmPDFSectionPagenum,
        )

    return output.confirmed


def adjust_section_pagenum(section, file_data, physical_offset, with_file=False):
    MAX_TRIES = 31  # radius 15
    page_idxs = list(range(len(file_data.pages if with_file else file_data)))
    page_idxs.reverse()

    # Note section.pagenum is probably 1-indexed
    search_start_i = section.pagenum - 1 + physical_offset
    tries = 0
    while len(page_idxs) > 0:
        # Search forward, then backward, with an increasing radius
        current_search_i = min(page_idxs, key=lambda x: abs(x - search_start_i))
        page_idxs.remove(current_search_i)

        print(
            f"Testing for section {section.title} (pagenum: {section.pagenum}) at pagenum {current_search_i + 1}"
        )

        confirmed = confirm_section_pagenum(
            section,
            file_data.pages[current_search_i]
            if with_file
            else file_data[current_search_i],
            data_type=("file" if with_file else "img"),
        )
        tries += 1

        if confirmed:
            section.pagenum = current_search_i
            return True, physical_offset + (current_search_i - search_start_i)

        if tries > MAX_TRIES:
            break

    section.pagenum = -1  # TODO: unclean hack
    return False, physical_offset


def parse_pdf_toc(fpath, reader, adjust_with_file=False):
    toc = parse_pdf_toc_naive(fpath)
    print("Parsed naive toc")

    # Adjust page numbers
    def flatten_toc(sections):
        flat_toc = []
        for section in sections:
            flat_toc.append(section)
            flat_toc.extend(flatten_toc(section.subsections))

        return flat_toc

    flat_naive_toc = flatten_toc(toc.sections)

    if not adjust_with_file:
        pdf_images = pdf2image.convert_from_path(fpath)

    current_physical_offset = 0
    for section in flat_naive_toc:
        confirmed, current_physical_offset = adjust_section_pagenum(
            section,
            reader if adjust_with_file else pdf_images,
            current_physical_offset,
            with_file=adjust_with_file,
        )
        if confirmed:
            print(f"Confirmed section {section.title} (at page {section.pagenum + 1})")
        else:
            print(f"Could not find section {section.title}")

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
                title=section.title,
                page_number=(section.pagenum if section.pagenum >= 0 else None),
                parent=parent,
            )
            apply_pdf_toc_sub(writer, section.subsections, parent=section_outline_item)

    apply_pdf_toc_sub(writer, toc.sections)
    writer.write(fpath)


def main():
    global fpath
    parser = argparse.ArgumentParser("autopdf")
    parser.add_argument("cmd")
    parser.add_argument("fpath", nargs="+")
    parser.add_argument(
        "--last-page",
        type=int,
        default=4,
        help="extract metadata from up to this page, 0-indexed",
    )
    parser.add_argument("--adjust-with-file", action="store_true")
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
            reader = PdfReader(fpath)
            if len(reader.outline) > 0 and not args.force:
                print(f"PDF has TOC (use --force): {fpath}; skipping")
                print()
                continue

            toc = parse_pdf_toc(fpath, reader, adjust_with_file=args.adjust_with_file)
            apply_pdf_toc(fpath, toc)
        else:
            err(f"Unknown cmd {args.cmd}")


if __name__ == "__main__":
    main()


# TODO:
# - Try higher reasoning with the naive method?
# - Allow configuring between img and file use in confirmation.
# - Output messages in stderr
# - Track expenses at end and if cancelled
# - Heuristic: increase reasoning level and try close ones in confirmation again
#   if getting real far away.
