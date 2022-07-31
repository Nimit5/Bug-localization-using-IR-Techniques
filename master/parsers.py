import glob
import os.path
from collections import OrderedDict

import javalang
import pygments
import xmltodict
from pygments.lexers import JavaLexer
from pygments.token import Token

# this class represents each bug report


class BugReport:

    __slots__ = ['summary', 'description', 'fixed_files',
                 'pos_tagged_summary', 'pos_tagged_description', 'stack_traces']

    def __init__(self, summary, description, fixed_files):
        self.summary = summary
        self.description = description
        self.fixed_files = fixed_files
        self.pos_tagged_summary = None
        self.pos_tagged_description = None
        self.stack_traces = None


# this class represents each source file
class SourceFile:

    __slots__ = ['all_content', 'comments', 'class_names', 'attributes', 'method_names',
                 'variables', 'file_name', 'pos_tagged_comments', 'exact_file_name', 'package_name']

    def __init__(self, all_content, comments, class_names, attributes,
                 method_names, variables, file_name, package_name):
        self.all_content = all_content
        self.comments = comments
        self.class_names = class_names
        self.attributes = attributes
        self.method_names = method_names
        self.variables = variables
        self.file_name = file_name
        self.exact_file_name = file_name[0]
        self.package_name = package_name
        self.pos_tagged_comments = None


class Parser:

    __slots__ = ['name', 'src', 'bug_repo']

    def __init__(self, project):
        self.name = project.name
        self.src = project.src
        self.bug_repo = project.bug_repo

    # parsing of XML format bug reports

    def report_parser(self):

        # converting XML bug repository to a dictionary
        with open(self.bug_repo) as xml_file:
            xml_dict = xmltodict.parse(
                xml_file.read(), force_list={'file', 'bug'})

        # iterating through bug reports and building their objects
        bug_reports = OrderedDict()

        for bug_report in xml_dict['bugrepository']['bug']:

            fixed_files = bug_report['fixedFiles']['file']
            if isinstance(fixed_files[0], dict):
                fixed_files = [f['#text'] for f in fixed_files]

            bug_reports[bug_report['@id']] = BugReport(
                bug_report['buginformation']['summary'],
                (bug_report['buginformation']['description']
                 if bug_report['buginformation']['description'] else ''),
                [os.path.normpath(path) for path in fixed_files],
            )

        return bug_reports
    # parsing source code directory of a program and collect their java files

    def src_parser(self):

        # getting the list of source files recursively from the source directory using regular expressions
        src_addresses = glob.glob(str(self.src) + '/**/*.java', recursive=True)

        # creating a java lexer instance for pygments.lex() method
        java_lexer = JavaLexer()

        src_files = OrderedDict()
        idx = 0
        # looping to parse each source file
        for src_file in src_addresses:
            # print(idx)
            idx += 1
            with open(src_file, encoding="ISO-8859-1") as file:
                src = file.read()

            # placeholder for different parts of a source file
            comments = ''
            class_names = []
            attributes = []
            method_names = []
            variables = []

            # source parsing
            parse_tree = None
            try:
                parse_tree = javalang.parse.parse(src)
                for path, node in parse_tree.filter(javalang.tree.VariableDeclarator):
                    if isinstance(path[-2], javalang.tree.FieldDeclaration):
                        attributes.append(node.name)
                    elif isinstance(path[-2], javalang.tree.VariableDeclaration):
                        variables.append(node.name)
            except:
                pass

            # lexically tokenize the source file
            lexed_src = pygments.lex(src, java_lexer)

            for i, token in enumerate(lexed_src):
                if token[0] in Token.Comment:
                    # removing the license comment
                    if i == 0 and token[0] is Token.Comment.Multiline:
                        src = src[src.index(token[1]) + len(token[1]):]
                        continue
                    comments += token[1]
                elif token[0] is Token.Name.Class:
                    class_names.append(token[1])
                elif token[0] is Token.Name.Function:
                    method_names.append(token[1])

            # get the package declaration if exists
            if parse_tree and parse_tree.package:
                package_name = parse_tree.package.name
            else:
                package_name = None

            if self.name == 'aspectj':
                src_files[os.path.relpath(src_file, start=self.src)] = SourceFile(
                    src, comments, class_names, attributes,
                    method_names, variables,
                    [os.path.basename(src_file).split('.')[0]],
                    package_name
                )
            else:
                # If source file has package declaration
                if package_name:
                    src_id = (package_name + '.' +
                              os.path.basename(src_file))
                else:
                    src_id = os.path.basename(src_file)

                src_files[src_id] = SourceFile(
                    src, comments, class_names, attributes,
                    method_names, variables,
                    [os.path.basename(src_file).split('.')[0]],
                    package_name
                )

        return src_files


def test():
    import datasets

    parser = Parser(datasets.zxing)
    x = parser.report_parser()
    d = parser.src_parser()

    src_id, src = list(d.items())[10]
    print("source_files: {}".format(src_id))
    print(src_id, src.exact_file_name, src.package_name)

    bug_id, bug = list(x.items())[10]
    print(bug_id, bug.fixed_files)


if __name__ == '__main__':
    test()
