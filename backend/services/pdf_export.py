import io
from xhtml2pdf import pisa

def generate_combined_pdf(html_docs: dict[str, str]) -> bytes:
    output = io.BytesIO()

    combined_html = ""

    for name, html in html_docs.items():
        combined_html += html
        combined_html += "<div style='page-break-after:always'></div>"

    pisa.CreatePDF(
        src=combined_html,
        dest=output
    )

    return output.getvalue()