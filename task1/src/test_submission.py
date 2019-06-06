from __future__ import print_function, division
import sys
import argparse
import zipfile
import os
import re


PY_VERSION = (3, 6)
PREDICTION_LEN = 20
PREDICTION_COUNT = 1176

zip_id = None

def check_exists(reader, path):
    entry_type = reader.get_type(path)

    if entry_type == Reader.NONE:
        append_error('File "{}" does not exist'.format(path))
    elif entry_type != Reader.FILE:
        append_error('Entry "{}" exists but is not a file'.format(path))
    else:
        return True

    return False


def check_directory(reader, path):
    entry_type = reader.get_type(path)

    if entry_type == Reader.NONE:
        append_error('Directory "{}" does not exist'.format(path))
    elif entry_type != Reader.DIRECTORY:
        append_error('Entry "{}" exists but is not a directry'.format(path))
    else:
        return True

    return False


def check_users(reader, path):
    if not check_exists(reader, path):
        return False

    with reader.open(path) as f:
        lines = f.read().splitlines()  # Use splitlines instead of readlines to keep newlines out

    error_count = 0
    ids = []

    for i, line in enumerate(lines):
        if len(line) == 0:
            append_error('{}:{}: Found empty line'.format(path, i + 1))
            error_count += 1
        else:
            m = re.match(r'^.+,?\s+(\d{3,9})$', line)

            if not m:
                append_error('{}:{}: "{}" does not match the required format'.format(path, i + 1, line))
                error_count += 1
            else:
                id = m.group(1)
                ids.append(id)

                check_id(id, '{}:{}'.format(path, i + 1))

    if zip_id is not None and zip_id not in ids:
        append_error('{}: ID from submission zip file ({}) not found'.format(path, zip_id))

    return error_count == 0


def check_predictions(reader, path):
    if not check_exists(reader, path):
        return False

    with reader.open(path) as f:
        lines = f.read().splitlines()  # Use splitlines instead of readlines to keep newlines out

    if len(lines) != PREDICTION_COUNT:
        append_error('{}: Incorrect line count ({}), should be {}'.format(path, len(lines), PREDICTION_COUNT))
        return False

    for i, line in enumerate(lines):
        if len(line) != PREDICTION_LEN:
            append_error('{}:{}: Incorrect line length ({}), should be {}'.format(path, i + 1, len(line), PREDICTION_LEN))
            return False

        for c in line:
            if c not in ['0', '1']:
                append_error('{}:{}: Invalid character: \'{}\' (should be \'0\' or \'1\')'.format(path, i + 1, c))
                return False

    return True


REQUIRED_ENTRIES = {
    'README.txt': check_exists,
    'USERS.txt': check_users,
    'project.pdf': check_exists,
    'oneof:Task': {
        'task1': {
            'src': {
                'requirements.txt': check_exists,
                'classifier.py': check_exists,
            },
        },
        'task2': {
            'src': {
                'requirements.txt': check_exists,
                'classifier.py': check_exists,
            }
        }
    }
}

class Reader(object):
    NONE = 'none'
    FILE = 'file'
    DIRECTORY = 'directory'

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        pass

    def entries(self):
        raise NotImplementedError('Must be overridden in a derived class')

    def get_type(self, path):
        raise NotImplementedError('Must be overridden in a derived class')

    def open(self, path, mode='r'):
        raise NotImplementedError('Must be overridden in a derived class')


class DirectoryReader(Reader):
    def __init__(self, path):
        self._path = path
        self._entries = self._build_entries(self._path)

    def entries(self):
        return self._entries

    def _build_entries(self, path):
        entries = []

        for entry in os.listdir(path):
            entry_path = os.path.join(path, entry)

            if os.path.islink(entry_path):
                append_warning('Encountered symbolic link at "{}"'.format(entry_path))

            if os.path.isdir(entry_path):
                entries.append(entry + '/')
                entries.extend((entry + '/' + child) for child in self._build_entries(entry_path))
            else:
                entries.append(entry)

        return entries

    def get_type(self, path):
        if path in self._entries:
            return self.DIRECTORY if path.endswith('/') else self.FILE
        elif path + '/' in self._entries:
            return self.DIRECTORY

        return self.NONE

    def open(self, path, mode='r'):
        if self.get_type(path) != self.FILE:
            raise Exception('Path is not a file')

        return open(os.path.join(self._path, os.path.normpath(path)), mode=mode)


class ZipReader(Reader):
    def __init__(self, path):
        self._zipfile = zipfile.ZipFile(path)

    def __enter__(self):
        self._zipfile.__enter__()
        self._entries = self._zipfile.namelist()
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self._zipfile.__exit__(exc_type, exc_value, exc_tb)

    def entries(self):
        return self._entries

    def get_type(self, path):
        if path in self._entries:
            return self.DIRECTORY if path.endswith('/') else self.FILE
        elif not path.endswith('/') and path + '/' in self._entries:
            return self.DIRECTORY

        return self.NONE

    def open(self, path, mode='r'):
        if self.get_type(path) != self.FILE:
            raise Exception('Path is not a file')

        file = self._zipfile.open(path, mode=mode)

        if 'b' not in mode and sys.version_info[0] == 3:
            # Wrap with TextIOWrapper to convert bytes to str (Python 3)
            if sys.version_info[1] <= 1:
                # Workaround for Python <= 3.1
                file.readable = lambda: True
                file.writable = lambda: False
                file.seekable = lambda: False
                file.read1 = file.read

            from io import TextIOWrapper
            file = TextIOWrapper(file)

        return file


def append_error(msg):
    global error_count
    error_count += 1
    print('ERROR:', msg)


def append_warning(msg):
    global warning_count
    warning_count += 1
    print('WARNING:', msg)


def check_python():
    version_str = '.'.join(str(c) for c in sys.version_info[:3])

    if sys.version_info[0] != PY_VERSION[0]:
        append_error('You are not using Python {req} (your Python version is {ver})'.format(req=PY_VERSION[0], ver=version_str))
        return

    if sys.version_info[1] > PY_VERSION[1]:
        append_warning('You are using Python {ver} which is newer than Python {req[0]}.{req[1]}. ' \
                       'Please make sure you are not using any features introduced ' \
                       'after Python 3.5.'.format(req=PY_VERSION, ver=version_str))


def check_required_entry(reader, base_name, name, value):
    found_entries = []

    if isinstance(value, dict):
        if name.startswith('oneof:'):
            name = name.split(':', 1)[1]
            existing_options = []

            for option_name, option_value in value.items():
                full_name = base_name + option_name
                option_type = reader.DIRECTORY if isinstance(option_value, dict) else reader.FILE
                entry_type = reader.get_type(full_name)

                if entry_type != Reader.NONE:
                    existing_options.append(full_name)

                    if entry_type != option_type:
                        append_error('"{}" exists but is not a {}'.format(full_name, option_type))

                    found_entries.extend(check_required_entry(reader, base_name, option_name, option_value))

            if len(existing_options) == 0:
                options = ', '.join('"{}"'.format(base_name + opt) for opt in value.keys())
                append_error(name + ': Could not find any of {}.'.format(options))
            elif len(existing_options) > 1:
                options = ', '.join('"{}"'.format(opt) for opt in existing_options)
                append_error(name + ': More than one exists: {}'.format(options))

        else:
            entry_name = base_name + name + '/'

            if check_directory(reader, entry_name):
                found_entries.append(entry_name)
                found_entries.extend(check_required_entries(reader, entry_name, value))
    else:
        if value(reader, base_name + name):
            found_entries = [base_name + name]

    return found_entries


def check_required_entries(reader, base_name, required_entries):
    found_entries = []

    for name, value in required_entries.items():
        found_entries.extend(check_required_entry(reader, base_name, name, value))

    return found_entries


def check_id(id, src):
    if len(id) != 9:
        append_warning('{}: ID number {} is shorter than 9 digits'.format(src, id))

    checksum = 0

    for i, c in enumerate(reversed(id)):
        n = (ord(c) - ord('0')) * ((i % 2) + 1)
        while n:
            checksum += n % 10
            n //= 10

    if checksum % 10 != 0:
        append_error('{}: ID number {} has an invalid checksum'.format(src, id))


def check_zip_name(name):
    global zip_id

    m = re.match(r'^(\d{3,9})\.zip$', name, re.IGNORECASE)
    if not m:
        append_error('Zip file name does not match requested format.')
    else:
        zip_id = m.group(1)

        check_id(zip_id, 'Zip file name ("{}")'.format(name))


def create_reader():
    if len(sys.argv) < 2:
        print('Usage: python {} <submission-file-or-directory-path>'.format(sys.argv[0]))
        sys.exit(1)

    input_path = sys.argv[1]

    if not os.path.exists(input_path):
        append_error('Input file or directory does not exist')
    elif os.path.isfile(input_path):
        check_zip_name(os.path.basename(input_path))
        return ZipReader(input_path)
    elif os.path.isdir(input_path):
        return DirectoryReader(input_path)


def main():
    global error_count, warning_count
    error_count = 0
    warning_count = 0

    reader_context = create_reader()

    check_python()

    if reader_context:
        with reader_context as reader:
            known = set(check_required_entries(reader, '', REQUIRED_ENTRIES))
            existing = set(reader.entries())
            unknown = list(existing - known)
            unknown.sort()

            hide_rules = {'task1/src/': 'hide_files_warn_pyc', 'task2/src/': 'hide_files_warn_pyc'}

            for path in unknown:
                hide_rule = None

                # Sort prefixes from longest to shortest so that we try longer ones first
                for prefix, rule in sorted(hide_rules.items(), key=lambda item: item[0], reverse=True):
                    if path.startswith(prefix):
                        hide_rule = rule
                        break

                # Completely ignore contents
                if hide_rule == 'hide':
                    continue

                if reader.get_type(path) == Reader.DIRECTORY:
                    append_warning('Unknown directory \"{}\", ignoring contents'.format(path))
                    hide_rules[path] = 'hide'
                else:
                    if hide_rule == 'hide_files_warn_pyc':
                        # Warn if found pyc file
                        if path.endswith('.pyc'):
                            append_warning('Found pyc file \"{}\". Do not submit pyc files if code is available.'.format(path))
                    else:
                        # Warn for any file
                        append_warning('Unknown file \"{}\"'.format(path))


    if error_count == 0 and warning_count == 0:
        print('No errors found')
    else:
        print('Total: {} errors, {} warnings'.format(error_count, warning_count))


if __name__ == '__main__':
    main()
