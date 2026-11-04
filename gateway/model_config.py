#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Shared stdlib-only structural ModelConfig identity reader."""
import os
from pathlib import Path
import re
import stat

CONFIG_LIMIT = 2 * 1024**2
DEPTH_LIMIT = 64


def _identity(info):
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def _read_config(path):
    path = Path(path)
    descriptor = None
    try:
        resolved = path.resolve(strict=True)
        descriptor = os.open(resolved, os.O_RDONLY | os.O_CLOEXEC | os.O_NONBLOCK | os.O_NOFOLLOW)
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError(f'{path}: expected regular protobuf config')
        if info.st_size > CONFIG_LIMIT:
            raise ValueError(f'{path}: protobuf config exceeds local byte limit')
        raw = os.pread(descriptor, info.st_size, 0)
        if len(raw) != info.st_size:
            raise ValueError(f'{path}: truncated protobuf config')
        if (_identity(info) != _identity(os.fstat(descriptor))
                or path.resolve(strict=True) != resolved
                or _identity(info) != _identity(resolved.stat())):
            raise ValueError(f'{path}: protobuf config changed during backend extraction')
        return raw.decode('utf-8')
    except (OSError, RuntimeError) as error:
        raise ValueError(f'{path}: cannot read protobuf config: {error}') from error
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _string_bytes(literal):
    """Decode protobuf C escapes to bytes before joining adjacent literals."""
    text = literal[1:-1]
    result = bytearray()
    position = 0
    escapes = {'a': 7, 'b': 8, 'f': 12, 'n': 10, 'r': 13, 't': 9, 'v': 11,
               "'": 39, '"': 34, '\\': 92, '?': 63}
    escape_pattern = re.compile(r"[0-7]{1,3}|x[0-9a-fA-F]{1,2}|u[0-9a-fA-F]{4}|U[0-9a-fA-F]{8}|[abfnrtv\\'\"?]")
    while position < len(text):
        char = text[position]
        position += 1
        if char != '\\':
            result.extend(char.encode('utf-8'))
            continue
        match = escape_pattern.match(text, position)
        if match is None:
            raise ValueError('invalid protobuf string escape')
        escaped = match.group()
        position += len(escaped)
        if escaped[0] in 'uU':
            result.extend(chr(int(escaped[1:], 16)).encode('utf-8'))
        elif escaped[0] == 'x':
            result.append(int(escaped[1:], 16))
        elif escaped[0] in '01234567':
            result.append(int(escaped, 8))
        else:
            result.append(escapes[escaped])
    return result


def read_model_config(path):
    """Read decoded top-level name/backend strings without runtime dependencies.

    This parses the text-format field/value structure, not ModelConfig's schema.
    Comments, nested fields and quoted lookalikes cannot select a backend. A
    missing identity field is None for legacy configs; ambiguous/malformed text fails.
    """
    text = _read_config(path)
    token_pattern = re.compile(
        r'''\s+|\#[^\r\n]*|(?P<string>"(?:[^"\\\r\n]|\\[^\r\n])*"|'(?:[^'\\\r\n]|\\[^\r\n])*')'''
        r'|(?P<identifier>[A-Za-z_][A-Za-z_0-9.]*)'
        r'|(?P<number>[+-]?(?:0[xX][0-9a-fA-F]+'
        r'|(?:[0-9]+(?:\.[0-9]*)?(?:[eE][+-]?[0-9]+)?'
        r'|\.[0-9]+(?:[eE][+-]?[0-9]+)?|(?i:inf(?:inity)?|nan))[fF]?))'
        r'|(?P<punctuation>[:{}<>\[\],;])')
    tokens, offset = [], 0
    while offset < len(text):
        match = token_pattern.match(text, offset)
        if match is None:
            raise ValueError(f'{path}: invalid protobuf token at character {offset}')
        if match.lastgroup:
            tokens.append((match.lastgroup, match.group()))
        offset = match.end()
    position = 0
    identity = {'name': None, 'backend': None}

    def peek():
        return tokens[position][1] if position < len(tokens) else None

    def value(depth, message_list=False):
        nonlocal position
        if depth > DEPTH_LIMIT:
            raise ValueError(f'{path}: protobuf nesting exceeds local limit')
        token = peek()
        if token in ('{', '<'):
            position += 1
            fields('}' if token == '{' else '>', depth+1)
        elif token == '[':
            position += 1
            if peek() != ']':
                while True:
                    if message_list and peek() not in ("{", "<"):
                        raise ValueError(f'{path}: expected protobuf message in colonless list')
                    value(depth+1)
                    if peek() != ',':
                        break
                    position += 1
                    if peek() == ']':
                        break
            if peek() != ']':
                raise ValueError(f'{path}: malformed protobuf list')
            position += 1
        elif position < len(tokens) and tokens[position][0] == 'string':
            result = bytearray()
            while position < len(tokens) and tokens[position][0] == 'string':
                result.extend(_string_bytes(tokens[position][1]))
                position += 1
            return result.decode('utf-8')
        elif position < len(tokens) and tokens[position][0] in ('identifier', 'number'):
            position += 1
        else:
            raise ValueError(f'{path}: missing protobuf field value')
        return None

    def fields(end=None, depth=0):
        nonlocal position
        while peek() is not None and peek() != end:
            if tokens[position][0] != 'identifier':
                raise ValueError(f'{path}: expected protobuf field name')
            name = peek()
            position += 1
            message_list = False
            if peek() == ':':
                position += 1
            elif peek() == '[':
                message_list = True
            elif peek() not in ('{', '<'):
                raise ValueError(f'{path}: missing colon for protobuf field {name}')
            parsed = value(depth, message_list=message_list)
            if depth == 0 and name in identity:
                if identity[name] is not None:
                    raise ValueError(f'{path}: duplicate top-level {name}')
                if not isinstance(parsed, str) or not parsed or '\x00' in parsed:
                    raise ValueError(f'{path}: {name} must be a nonempty protobuf string without NUL')
                identity[name] = parsed
            if peek() in (',', ';'):
                position += 1
        if end is not None:
            if peek() != end:
                raise ValueError(f'{path}: unclosed protobuf message')
            position += 1
    fields()
    return identity


if __name__ == '__main__':
    # Shared watcher entry point; no model/runtime packages are imported.
    import sys
    try:
        if len(sys.argv) not in (2, 3) or (len(sys.argv) == 3 and sys.argv[2] not in ('name', 'backend')):
            raise ValueError('expected a config.pbtxt path and optional name/backend field')
        field = sys.argv[2] if len(sys.argv) == 3 else 'backend'
        print(read_model_config(sys.argv[1])[field] or '')
    except (ValueError, SyntaxError, UnicodeError) as error:
        print(f'backend extraction failed: {error}', file=sys.stderr)
        raise SystemExit(1)
