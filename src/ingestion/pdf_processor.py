
import os
import glob
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.config.parser import ConfigParser
from marker.output import MarkdownOutput

# Define paths for raw PDFs and the directory for extracted markdown
RAW_PDFS_DIR = "data/raw_pdfs"
EXTRACTED_MARKDOWN_DIR = "data/extracted_markdown"

def process_pdfs():
    """
    Converts all PDF files in the RAW_PDFS_DIR to markdown format
    and saves them in the EXTRACTED_MARKDOWN_DIR.
    Skips files that have already been converted.
    """
    # Ensure the output directory exists
    os.makedirs(EXTRACTED_MARKDOWN_DIR, exist_ok=True)

    # Find all PDF files in the raw_pdfs directory
    pdf_files = glob.glob(os.path.join(RAW_PDFS_DIR, "*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in {RAW_PDFS_DIR}")
        return

    print(f"Found {len(pdf_files)} PDF(s) to process.")

    # In-memory cache for models
    model_dict = create_model_dict()

    for pdf_path in pdf_files:
        # Determine the output path
        base_name = os.path.basename(pdf_path)
        file_name_without_ext = os.path.splitext(base_name)[0]
        output_path = os.path.join(EXTRACTED_MARKDOWN_DIR, f"{file_name_without_ext}.md")

        # Skip if the file has already been converted
        if os.path.exists(output_path):
            print(f"Skipping {pdf_path}, already converted.")
            continue

        print(f"Processing: {pdf_path}")

        # Basic config for markdown output
        # marker-pdf can be used with local models by default
        config = {
            "output_format": "markdown",
        }
        config_parser = ConfigParser(config)

        # Set up the converter
        converter = PdfConverter(
            config=config_parser.generate_config_dict(),
            artifact_dict=model_dict,
            processor_list=config_parser.get_processors(),
            renderer=config_parser.get_renderer(),
            llm_service=config_parser.get_llm_service()
        )

        # Convert the PDF
        rendered_output: MarkdownOutput = converter(pdf_path)

        # Save the markdown content
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(rendered_output.markdown)

        print(f"Successfully converted {pdf_path} to {output_path}")

if __name__ == "__main__":
    process_pdfs()
