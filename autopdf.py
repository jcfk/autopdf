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
import time
from pathlib import Path
from io import BytesIO

import pdf2image
from openai import OpenAI
from pypdf import PdfReader, PdfWriter
from pydantic import BaseModel


GPT_5_4_NANO = {"id": "gpt-5.4-nano", "cost_in_1m": 0.2, "cost_out_1m": 1.25}
SELECTED_MODEL = GPT_5_4_NANO

args = None
fpath = None
usage = {"tok_in": 0, "tok_out": 0, "est_cost": 0}


class ParsedPDFMetadata(BaseModel):
    # TODO the first three should be optional
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


class ParsedPDFIndexFlatEntry(BaseModel):
    name: str
    page_number: int


class ParsedPDFIndexEntry(BaseModel):
    name: str
    page_numbers: list[int]


class ParsedPDFIndex(BaseModel):
    entries: list[ParsedPDFIndexEntry]


class ParsedPDFPageNumbers(BaseModel):
    pagenums: list[int | None]


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
        model=SELECTED_MODEL["id"],
        reasoning={"effort": reasoning},
        text_format=response_format,
        input=[
            {
                "role": "user",
                "content": message_content,
            }
        ],
    )

    tok_in = response.usage.input_tokens
    tok_out = response.usage.output_tokens
    usage["tok_in"] += tok_in
    usage["tok_out"] += tok_out
    usage["est_cost"] += (
        tok_in / 1000000 * SELECTED_MODEL["cost_in_1m"]
        + tok_out / 1000000 * SELECTED_MODEL["cost_out_1m"]
    )

    # ???
    return response.output_parsed


def get_pdf_slice_as_data(reader, first_page=None, last_page=None):
    with tempfile.TemporaryFile() as f:
        writer = PdfWriter()
        i = first_page if first_page is not None else 0
        limit = last_page if last_page is not None else len(reader.pages) - 1
        # inclusive, 0-indexed
        pages = reader.pages[i : limit + 1]
        for page in pages:
            writer.add_page(page)
        writer.write(f)
        f.seek(0)
        # breakpoint()
        return f.read(), len(pages)


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
    with tempfile.TemporaryFile() as f:
        reader = PdfReader(fpath)
        writer = PdfWriter()
        if args.make_toc_last_page:
            for i in range(min(len(reader.pages), args.make_toc_last_page)):
                writer.add_page(reader.pages[i])
        else:
            writer.append_pages_from_reader(reader)
        writer.write(f)

        f.seek(0)
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
    MAX_TRIES = 21  # radius 15
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
            f'Testing for section "{section.title}" (pagenum: {section.pagenum}) at pagenum {current_search_i + 1}'
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
    print("Preliminary naive toc:")
    pprint_toc(toc)

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


def parse_pdf_index_naive(reader, first_page):
    page_count = len(reader.pages)
    data, _ = get_pdf_slice_as_data(reader, first_page)
    data_b64 = base64.b64encode(data).decode("utf-8")
    file_data = {"fname": fpath.name, "data_b64": data_b64}

    message = (
        "Extract the  index from the  provided pages  of the book.  Format your"
        " response as  a json  object, where  key 'entries' is  a list  of index"
        " entries. Index  entries are also json  objects, where key 'name'  is a"
        " string with the name of the entry  and key 'page_numbers' is a list of"
        " integers corresponding to the provided page numbers for that entry."
        " If an  index entry has sub-entries,  propagate the name of  the parent"
        " entry across  the sub-entries  and save  each sub  entry individually,"
        " with its  page numbers.  For example,  the entry  'Kolmogorov's' might"
        " have sub-entries  'continuity criterion  357', 'cycle  condition 298',"
        " 'maximal inequality 79',  etc, each with their own  page numbers. Save"
        " entries 'Kolmogorov's  continuity criterion  357', etc. If  the parent"
        " entry has its own page number,  save the parent entry by itself, also."
        " For  example, the  entry 'Cauchy-Schwarz  inequality 105'  may have  a"
        " typeset  child  'conditional  179'.  Then  save  both  'Cauchy-Schwarz"
        " inequality  105'  and  'Cauchy-Schwarz  inequality  conditional  179'."
        " Determine whether an entry has sub-entries by examining their relative"
        " type-setting."
    )

    output = llm_helpful_assistant(
        message,
        reasoning="medium",
        media_type="file",
        media_data=file_data,
        response_format=ParsedPDFIndex,
    )

    return output


def read_pagenum_db(fpath):
    with open(fpath, "r") as f:
        book_pnum_to_pdf_pnum = {}
        for line in f:
            line = line.rstrip()
            pdf_pnum, book_pnum = line.split(" ")
            pdf_pnum = int(pdf_pnum)
            try:
                book_pnum = int(book_pnum)
                book_pnum_to_pdf_pnum[book_pnum] = pdf_pnum
            except:
                pass

        return book_pnum_to_pdf_pnum


def parse_pdf_index(
    fpath, first_page, validation_method, page_offset=None, pnum_db_fpath=None
):
    reader = PdfReader(fpath)
    index = parse_pdf_index_naive(reader, first_page)

    flat_index = ParsedPDFIndex(entries=[])
    for entry in index.entries:
        name = entry.name
        dupe_entries = len(entry.page_numbers) > 1

        i = 1
        for pnum in entry.page_numbers:
            flat_entry = ParsedPDFIndexFlatEntry(
                name=f"{name} ({i})" if dupe_entries else name, page_number=pnum
            )
            flat_index.entries.append(flat_entry)
            i += 1

    if validation_method == "radial_adjust":
        pass
    elif validation_method == "naive_offset":
        for entry in flat_index.entries:
            entry.page_number += page_offset
    elif validation_method == "pagenum_db":
        db = read_pagenum_db(pnum_db_fpath)
        for entry in flat_index.entries:
            entry.page_number = db[entry.page_number]

    return flat_index


def print_index(index, fpath):
    with open(fpath, "w") as f:
        for entry in index.entries:
            name = entry.name
            pnum = entry.page_number
            f.write(f"{pnum} {name}\n")


def parse_pagenums(fpath):
    BATCH_SIZE_LIMIT = 5

    reader = PdfReader(fpath)
    pdf_page_to_book_page = []

    # expected_book_pnum = 0
    for i in range(0, len(reader.pages), BATCH_SIZE_LIMIT):
        data, batch_size = get_pdf_slice_as_data(
            reader, first_page=i, last_page=i + BATCH_SIZE_LIMIT - 1
        )
        data_b64 = base64.b64encode(data).decode("utf-8")
        file_data = {"fname": fpath.name, "data_b64": data_b64}

        pdf_pnums = [i + d + 1 for d in range(batch_size)]

        message = (
            f"Extract  the integer  page number  from the  {batch_size} provided"
            " book pages. Format  your response as a json object,  where the key"
            " 'pagenums'  is   a  list[int  |  None]   of  length  {batch_size},"
            " corresponding to the pages. Each element is either an integer page"
            " number for  that page, or  None, if for  example when there  is no"
            " page number shown, or when the  page is numbered in roman numerals"
            " or another non-integer  scheme. Do not return a  number unless the"
            " page has an integer book page number."
            " The book  may be a  scan, in which case  pages may be  missing. In"
            " this case, page numbers returned need not always be consecutive."
            # The book  may be  a scan,  with additional  pages may  be inserted
            # between adjacent book  pages. In this case, use  None to represent
            # these non-book pages added between original book pages.
            " The book may  neglect numbering certain pages,  like chapter title"
            " pages. In this case, infer the page number from the following page"
            " numbers."
            # " You  may  take  into  account   that  the  previous  page  number  was"
            # f" {expected_book_pnum-1},    so   the    expected    page   number    is"
            # f" {expected_book_pnum}."
        )

        # breakpoint()

        output = llm_helpful_assistant(
            message,
            reasoning="medium",
            media_type="file",
            media_data=file_data,
            response_format=ParsedPDFPageNumbers,
        )

        for pdf_pnum, book_pnum in zip(pdf_pnums, output.pagenums):
            if book_pnum is not None:
                print(f"PDF page {pdf_pnum}, found page number {book_pnum}")
            else:
                print(f"PDF page {pdf_pnum}, found page number {book_pnum}")
            pdf_page_to_book_page.append((pdf_pnum, book_pnum))

        # if book_pnum != expected_book_pnum:
        #     for j in range(expected_book_pnum, book_pnum):
        #         book_page_to_pdf_page.append((j, "?"))
        # expected_book_pnum = book_pnum + 1

    return pdf_page_to_book_page


def save_pagenum_db(pdf_page_to_book_page, fpath):
    with open(fpath, "w") as f:
        for pdf_pnum, book_pnum in pdf_page_to_book_page:
            f.write(f"{pdf_pnum} {book_pnum}\n")


def main():
    global args, fpath
    parser = argparse.ArgumentParser("autopdf")
    parser.add_argument("cmd")
    parser.add_argument("fpath", nargs="+")
    parser.add_argument(
        "--rename-last-page",
        type=int,
        default=9,
        help="extract metadata from up to this page, 0-indexed",
    )
    parser.add_argument(
        "--make-toc-last-page",
        type=int,
        default=None,
        help="parse frontmatter for toc up to this page, 1-indexed",
    )
    parser.add_argument("--make-toc-adjust-with-file", action="store_true")
    parser.add_argument("--make-toc-force", action="store_true")
    parser.add_argument(
        "--parse-index--first-page",
        type=int,
        default=-25,
        help="first page to consider in extracting index, 1-indexed",
    )
    parser.add_argument("--parse-index--output")
    # This should be the 1-indexed physical page of book page 1, minus 1.
    parser.add_argument("--parse-index--validation-method")
    parser.add_argument("--parse-index--page-offset", type=int)
    parser.add_argument("--parse-index--pagenum-db")
    parser.add_argument("--make-pagenum-db--output")
    args = parser.parse_args()

    start_time = time.time()

    for fpath in args.fpath:
        fpath = Path(fpath)

        if args.cmd == "rename":
            parsed_metadata = parse_pdf_metadata(fpath, last_page=args.rename_last_page)
            if parsed_metadata.error:
                print(f"Error processing {fpath}; skipping")
                print()
                continue

            new_fpath = rename_pdf(fpath, parsed_metadata)
        elif args.cmd == "make-toc":
            reader = PdfReader(fpath)
            if len(reader.outline) > 0 and not args.make_toc_force:
                print(f"PDF has TOC (use --force): {fpath}; skipping")
                print()
                continue

            toc = parse_pdf_toc(
                fpath, reader, adjust_with_file=args.make_toc_adjust_with_file
            )
            apply_pdf_toc(fpath, toc)
        elif args.cmd == "parse-index":
            # TODO also allow user to provide --parse-index--pagenum-db
            if not args.parse_index__validation_method:
                err(f"Must provide --parse-index--validation-method")

            index = parse_pdf_index(
                fpath,
                args.parse_index__first_page - 1,
                args.parse_index__validation_method,
                page_offset=args.parse_index__page_offset,
                pnum_db_fpath=args.parse_index__pagenum_db,
            )

            if args.parse_index__output:
                index_fpath = args.parse_index_output
            else:
                index_fpath = f"{fpath}.index.autopdf"
            print_index(index, index_fpath)
        elif args.cmd == "make-pagenum-db":
            pdf_page_to_book_page = parse_pagenums(fpath)

            if args.make_pagenum_db__output:
                db_fpath = args.make_pagenum_db__output
            else:
                db_fpath = f"{fpath}.pagenum_db.autopdf"
            save_pagenum_db(pdf_page_to_book_page, db_fpath)
        else:
            err(f"Unknown cmd {args.cmd}")

    elapsed_time = time.time() - start_time
    print(f"Total time: {elapsed_time:.2f}s")
    print(
        f"Total usage: tokens in: {usage['tok_in']}, tokens out: {usage['tok_out']}, est cost: ${usage['est_cost']:.6f}"
    )


if __name__ == "__main__":
    main()


# TODO:
# - Try higher reasoning with the naive method?
# - Output messages in stderr
# - Output expenses even if cancelled
# - Numbered section titles in the outline
# - Index extraction would be really cool. Parse out the index, and run the same
#   physical page confirmation procedure.
# - Close TOC by default in firefox?
# - Ignore descriptive subheadings in frontmatter TOCs
# - A lot of problems would be solved by simply having an correspondence of book
#   pnums  to physical  pnums. This  would need  to be  stored in  some kind  of
#   system-wide database.
