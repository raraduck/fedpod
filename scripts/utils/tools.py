import os
import argparse
import re
import logging

def parse_1d_str_list(string):
    try:
        # 외부 대괄호 제거 및 공백 제거
        string = string.strip()
        # 외부 대괄호 확인
        if string[0] != '[' or string[-1] != ']':
            raise ValueError("Input must be wrapped in outer brackets.")
        # string = string[1:-1]
        # 외부 대괄호 제거
        inner_string = string[1:-1].strip()
        if not inner_string:
            raise ValueError("Inner content cannot be empty.")
        # 내부 리스트 파싱
        return [el.strip() for el in inner_string.split(',')]
    except Exception as e:
        # 로깅: 에러 발생 시 로그에 기록
        # logger.error(f"Failed to parse 1D list: {str(e)}")
        raise argparse.ArgumentTypeError("1D list must be in the format [str, str, ...]")


def parse_1d_int_list(string):
    try:
        # 외부 대괄호 제거 및 공백 제거
        string = string.strip()
        # 외부 대괄호 확인
        if string[0] != '[' or string[-1] != ']':
            raise ValueError("Input must be wrapped in outer brackets.")
        # string = string[1:-1]
        # 외부 대괄호 제거
        inner_string = string[1:-1].strip()
        if not inner_string:
            raise ValueError("Inner content cannot be empty.")
        # 내부 리스트 파싱
        return [int(el) for el in inner_string.split(',')]
    except Exception as e:
        # 로깅: 에러 발생 시 로그에 기록
        # logger.error(f"Failed to parse 1D list: {str(e)}")
        raise argparse.ArgumentTypeError("1D list must be in the format [int, int, ...]")


def parse_1d_float_list(string):
    try:
        # 외부 대괄호 제거 및 공백 제거
        string = string.strip()
        # 외부 대괄호 확인
        if string[0] != '[' or string[-1] != ']':
            raise ValueError("Input must be wrapped in outer brackets.")
        # string = string[1:-1]
        # 외부 대괄호 제거
        inner_string = string[1:-1].strip()
        if not inner_string:
            raise ValueError("Inner content cannot be empty.")
        # 내부 리스트 파싱
        return [float(el) for el in inner_string.split(',')]
    except Exception as e:
        # 로깅: 에러 발생 시 로그에 기록
        # logger.error(f"Failed to parse 1D list: {str(e)}")
        raise argparse.ArgumentTypeError("1D list must be in the format [float, float, ...]")


def parse_2d_int_list(string):
    try:
        # 외부 대괄호 제거 및 공백 제거
        string = string.strip()
        # 외부 대괄호 확인
        if string[0] != '[' or string[-1] != ']':
            raise ValueError("Input must be wrapped in outer brackets.")
        # string = string[1:-1]
        # 외부 대괄호 제거
        inner_string = string[1:-1].strip()
        if not inner_string:
            raise ValueError("Inner content cannot be empty.")

        # 내부 리스트 파싱
        pattern = r'\[(.*?)\]'
        matches = re.findall(pattern, inner_string)
        if not matches:
            raise ValueError("No valid inner list found.")
        parsed_lists = [list(map(int, match.split(','))) for match in matches]
        # 각 리스트의 길이 확인 (2D 리스트 형식 강제)
        if not all(len(lst) >= 2 for lst in parsed_lists):  # 예를 들어 각 리스트의 길이를 2로 제한
            raise ValueError("Each sublist must contain two integers.")
        return parsed_lists
    except Exception as e:
        # 로깅: 에러 발생 시 로그에 기록
        # logger.error(f"Failed to parse 2D list of label_groups: {str(e)}")
        raise argparse.ArgumentTypeError("2D list must be in the format [[int, int], [int, int], ...]")


def parse_args_to_dict(args):
    args_dict = {}
    args = [item.strip() for item in args if item not in ['\\']]
    it = iter(args)
    while True:
        try:
            key = next(it)
            if key.startswith('-'):
                try:
                    value = next(it)
                    if value.startswith('-'):
                        # 값 대신 키가 나왔으므로 이전 키의 값은 'true'
                        args_dict[key] = True
                        # 현재의 키를 다시 처리하기 위해 이터레이터를 한 단계 뒤로
                        it = iter([f"{value}"] + list(it))
                    else:
                        args_dict[key] = value
                except StopIteration:
                    # 다음 값이 없으면 'true' 사용
                    args_dict[key] = True
            else:
                raise ValueError(f"Argument {key} does not start with '-'")
        except StopIteration:
            break
    return args_dict