from pathlib import Path

from PyPDF2 import PdfFileWriter, PdfFileReader
import click


def grab_pages(original, from_, to):
    """
    Extract pages from original ReaderObject and produce new smaller reader
    
    Parameters
    ==========
    original: PdfFileReader
        reader object to extract pages from
    from_: int
        start page
    to: int
        end page (inclusive)
    """
    smaller = PdfFileWriter()
    for page in range(from_, to + 1): # inclusive
        smaller.insertPage(original.getPage(page), index=page)

    _import_bookmarks(original, smaller, from_, to)
    
    return smaller


def _import_bookmarks(original, writer, start, end, _outlines=None, _parent=None):
    if _outlines is None:
        _outlines = original.getOutlines()
    
    next_parent = _parent
    
    for outline in _outlines:
        if isinstance(outline, dict): # top level outline
            page = original.getDestinationPageNumber(outline)
        
            if not start <= page <= end:
                continue # out of range
        
            title = outline['/Title']
            next_parent = writer.addBookmark(title, page - start, parent=_parent)
        else:
            _import_bookmarks(original, writer, start, end, outline, _parent=next_parent) # parent previous top level outline
        
        
    

@click.group()
def cli():
    pass
    


@cli.command()
@click.argument('file', type=click.Path(exists=True))
@click.argument('start', type=int)
@click.argument('end', type=int)
def get_pages(start, end, file):
    start += -1 # 0 indexing in place
    end += -1
    
    file = Path(file)
    original = PdfFileReader(file.open('rb'))
    
    click.echo("Loaded file with {} pages, selecting {} to {}".format(original.getNumPages(), start + 1, end + 1))

    
    writer = grab_pages(original, start, end)
        
    new_file = file.with_name(file.name[:-4] + '- pages {} to {}.pdf'.format(start + 1, end + 1))
    with open(new_file, 'wb') as out:
        writer.write(out)
    
    click.echo("Successfully written to: {}".format(new_file.as_posix()))

if __name__ == '__main__':
    cli()


    
