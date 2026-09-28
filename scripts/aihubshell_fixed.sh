#!/bin/bash
# AI Hub 공식 CLI(aihubshell)의 패치 버전.
# 원본: https://api.aihub.or.kr/api/aihubshell.do (aihubshell version 25.09.19 v0.6)
#
# 2026-09-27~28 SafeWatch_AI 세션에서 도로주행영상 원천데이터(14GB, 14개
# 조각) 다운로드를 반복 시도하며 발견한 버그 2개를 고쳤다. OS와 무관한
# 스크립트 로직 결함이라 어디서 받든 재현된다 (docs/sprint-plan.md
# "aihubshell 자체 버그 2개" 참고).
#
#   [버그 1] curl_exit_code=$?가 curl이 아니라 그 앞 echo의 종료코드를
#   읽어와 항상 0(성공)으로 나옴 — 다운로드가 중간에 끊겨도 계속 진행됨.
#   → curl 실행 직후에 바로 캡처하도록 수정.
#
#   [버그 2] merge_parts()의 sort -t'.' -k2V가 "이름.zip.partNNNN"에서
#   항상 "zip"인 2번째 필드로 정렬해 사실상 정렬이 안 됨 — 조각이 여러
#   개(대용량 파일)면 순서가 뒤섞여 병합되어 파일이 깨짐.
#   → part 뒤 숫자만 뽑아 진짜 숫자로 정렬하도록 수정.
#
# 사용법은 원본과 동일:
#   ./aihubshell_fixed.sh -mode l -datasetkey <dataSetSn>
#   ./aihubshell_fixed.sh -aihubapikey <키> -mode d -datasetkey <dataSetSn> -filekey <filekey1,filekey2,...>

echo "=========================================="
echo "aihubshell version 25.09.19 v0.6 (SafeWatch 패치: 버그 2개 수정)"
echo "=========================================="

VER="0.6"
BASE_URL="https://api.aihub.or.kr"
LOGIN_URL="$BASE_URL/api/keyValidate.do"
BASE_DOWNLOAD_URL="$BASE_URL/down/$VER"
MANUAL_URL="$BASE_URL/info/api.do"
BASE_FILETREE_URL="$BASE_URL/info"
DATASET_URL="$BASE_URL/info/dataset.do"
DATAPCKAGE_URL="$BASE_URL/info/datapckage.do"
BASE_PCKAGETREE_URL="$BASE_URL/info/pckage"
BASE_PCKAGE_DOWNLOAD_URL="$BASE_URL/down/pckage/$VER"

print_usage() {
    manual=$(curl -s "$MANUAL_URL")
    echo $manual | grep -oP '"SJ":"\K[^"]+' | head -n 1
    echo -e '\n'
    echo -e "COMMAND\t\t OPTION\t\t\t\t DETAIL"
    echo "$manual" | awk -F'"' -v RS='},' '
    /ENGL_CMGG/ && /KOREAN_CMGG/ && /DETAIL_CN/ {
        for(i=1; i<=NF; i++) {
            if($i == "ENGL_CMGG") engl=$(i+2)
            if($i == "KOREAN_CMGG") korean=$(i+2)
            if($i == "DETAIL_CN") {
                detail=$(i+2)
                gsub(/\\n/, "\n", detail)
                gsub(/\\t/,"\t\t", detail)
                gsub(/\\/,"", detail)
            }
        }
        printf "%-10s\t %-15s\t|\t %s\n\n", engl, korean, detail
    }'
}

if [[ "$1" == "-help" ]]; then
    print_usage
    exit 0
fi

while [[ "$#" -gt 0 ]]; do
    case $1 in
        -aihubapikey) aihubapikey="$2"; shift ;;
        -mode)
            mode="$2";
            if [[ "$mode" == "l" ]]; then
                if [[ "$3" =~ ^[0-9]+$ ]]; then
                    datasetkey="$3";
                    shift;
                fi
            elif [[ "$mode" == "pl" ]]; then
                if [[ "$3" =~ ^[0-9]+$ ]]; then
                    datapckagekey="$3";
                    shift;
                fi
            fi
            shift;
            ;;
        -datasetkey) datasetkey="$2"; shift ;;
        -filekey) filekeys="$2"; shift ;;
        -datapckagekey) datapckagekey="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

if [[ "$mode" == "d" && -z "$datasetkey" ]]; then
    echo "Error: datasetkey is required when mode is 'd'";
    exit 1;
elif [[ "$mode" == "pd" && -z "$datapckagekey" ]]; then
    echo "Error: datapckagekey is required when mode is 'pd'";
    exit 1;
fi

aihubapikey=${aihubapikey:-$AIHUB_APIKEY}
filekeys=${filekeys:-"all"}

cleanup() {
    if [ -e "download.tar" ]; then
        rm "download.tar"
        echo -e "\n다운로드가 중단되었습니다."
    fi
    exit 1
}

down_func() {
    local DOWN_URL="$1";

    # [SafeWatch 패치 - 버그 1] curl 종료 코드를 curl 실행 직후 바로 캡처한다.
    http_response=$(curl -L -C - -o "download.tar" -H "apikey:$aihubapikey" -w "\n%{http_code}" "$DOWN_URL?fileSn=$filekeys")
    curl_exit_code=$?
    http_status=$(echo "$http_response" | tail -n1)
    http_body=$(echo "$http_response" | sed '$d')

    if [ "$http_status" -eq 200 ] && [ $curl_exit_code -eq 0 ]; then
        echo "Request successful with HTTP status $http_status."
        echo "Download successful."
        tar -xvf download.tar

        echo '잠시 기다려 주세요 병합중 입니다. '
        merge_parts() {
            local target_dir="$1"

            for prefix in $(ls "$target_dir" | grep '.*\.part[0-9]*$' | sed 's/\(.*\)\.part[0-9]*$/\1/' | sort -u); do
                escaped_prefix=$(printf '%q' "$prefix")
                echo "Merging $prefix in $target_dir"

                # [SafeWatch 패치 - 버그 2] part 뒤 숫자(바이트 오프셋)만 뽑아
                # 진짜 숫자로 정렬한다. 원본은 "이름.zip.partNNNN"을 '.'으로
                # 나눈 2번째 필드("zip", 모든 조각이 동일)로 정렬해 조각이
                # 여러 개일 때 순서가 뒤섞였다.
                : > "${target_dir}/${prefix}"
                find "${target_dir}" -name "${escaped_prefix}.part*" -print0 |
                    while IFS= read -r -d '' f; do
                        printf '%s\t%s\n' "${f##*.part}" "$f"
                    done | sort -n -k1,1 | cut -f2- |
                    while IFS= read -r f; do
                        cat "$f" >> "${target_dir}/${prefix}"
                    done

                rm "${target_dir}/${prefix}".part*
            done
        }

        find . -type d | while read -r dir; do
            if [[ ! -z $(find "$dir" -maxdepth 1 -name '*.part*') ]]; then
                merge_parts "$dir"
            fi
        done

        echo '병합이 완료 되었습니다. '
        rm download.tar
    else
        echo "Download failed with HTTP status $http_status, curl exit code $curl_exit_code."
        echo "Error msg:"
        cat download.tar 2>/dev/null
        rm -f download.tar
    fi
}

case $mode in
    d)
        trap cleanup SIGINT

        if [ -e "download.tar" ]; then
            TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
            mv "download.tar" "download_$TIMESTAMP.tar"
            echo "msg : download.tar 파일이 존재하여 download_$TIMESTAMP.tar로 백업하였습니다."
        fi
        DOWNLOAD_URL="$BASE_DOWNLOAD_URL/$datasetkey.do"
        down_func $DOWNLOAD_URL
        ;;
    l)
        if [[ $datasetkey ]]; then
            FILETREE_URL="$BASE_FILETREE_URL/$datasetkey.do"
            echo "Fetching file tree structure..."
            file_tree=$(curl -s "$FILETREE_URL")
            echo "$file_tree"
        else
            echo "Fetching dataset information..."
            dataset_info=$(curl -s "$DATASET_URL")
            echo "$dataset_info"
        fi
        ;;
    pl)
        if [[ $datapckagekey ]]; then
            PCKAGETREE_URL="$BASE_PCKAGETREE_URL/$datapckagekey.do"
            echo "Fetching datapckage file tree structure..."
            pckage_tree=$(curl -s "$PCKAGETREE_URL")
            echo "$pckage_tree"
        else
            echo "Fetching datapckage information..."
            datapckage_info=$(curl -s "$DATAPCKAGE_URL")
            echo "$datapckage_info"
        fi
        ;;
    pd)
        trap cleanup SIGINT

        if [ -e "download.tar" ]; then
            TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
            mv "download.tar" "download_$TIMESTAMP.tar"
            echo "msg : download.tar 파일이 존재하여 download_$TIMESTAMP.tar로 백업하였습니다."
        fi
        PCKAGE_DOWNLOAD_URL="$BASE_PCKAGE_DOWNLOAD_URL/$datapckagekey.do"
        down_func $PCKAGE_DOWNLOAD_URL
        ;;
    *)
        echo "Invalid mode. Please use 'pd','pl' for datapckage list,'d', 'l' for dataset list, 'l [datasetkey]' for file tree"
        exit 1
        ;;
esac
