#!/bin/bash
# ============================================================
# Nanobot + FastCode 智能启动脚本
# 自适应检测环境状态，自动配置、构建、启动
# 飞书 <-> Nanobot <-> FastCode 全链路通信
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
COMPOSE_FILE="$PROJECT_ROOT/docker-compose.yml"
NANOBOT_CONFIG="$PROJECT_ROOT/nanobot_config.json"
HOME_NANOBOT_CONFIG="$HOME/.nanobot/config.json"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# docker compose wrapper
dc() {
    docker compose -f "$COMPOSE_FILE" "$@"
}

print_banner() {
    echo ""
    echo "╔══════════════════════════════════════════════╗"
    echo "║   Nanobot + FastCode  智能启动脚本           ║"
    echo "║   飞书 <-> Nanobot <-> FastCode              ║"
    echo "╚══════════════════════════════════════════════╝"
    echo ""
}

# ============ 检查 Docker 环境 ============
check_docker() {
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}[Docker] 未检测到 Docker，请先安装 Docker Desktop${NC}"
        echo "   下载地址: https://www.docker.com/products/docker-desktop"
        exit 1
    fi

    if ! docker info &> /dev/null 2>&1; then
        echo -e "${RED}[Docker] Docker 服务未运行，请先启动 Docker Desktop${NC}"
        exit 1
    fi
}

# ============ 检测容器运行状态 ============
# 返回: "running" / "stopped" / "none"
get_container_state() {
    local service_name="$1"
    local container_id
    container_id=$(dc ps -q "$service_name" 2>/dev/null)

    if [ -z "$container_id" ]; then
        echo "none"
        return
    fi

    local state
    state=$(docker inspect -f '{{.State.Status}}' "$container_id" 2>/dev/null || echo "none")
    if [ "$state" = "running" ]; then
        echo "running"
    else
        echo "stopped"
    fi
}

# ============ 检测 Docker 镜像是否存在 ============
images_exist() {
    local fc_img nb_img
    fc_img=$(docker images --format '{{.Repository}}:{{.Tag}}' 2>/dev/null | grep -c "fastcode" || true)
    nb_img=$(docker images --format '{{.Repository}}:{{.Tag}}' 2>/dev/null | grep -c "nanobot" || true)
    [ "$fc_img" -gt 0 ] && [ "$nb_img" -gt 0 ]
}

# ============ 自动配置 nanobot_config.json ============
auto_configure_nanobot() {
    echo -e "${BLUE}[Config] 检查 Nanobot 配置...${NC}"

    if [ -f "$NANOBOT_CONFIG" ]; then
        # 配置已存在，检查是否有占位符
        if grep -q '"your_feishu_app_id"' "$NANOBOT_CONFIG" 2>/dev/null; then
            echo -e "${YELLOW}   nanobot_config.json 中飞书配置仍为占位符${NC}"
            # 尝试从 ~/.nanobot/config.json 自动填充
            if [ -f "$HOME_NANOBOT_CONFIG" ]; then
                echo -e "${CYAN}   检测到 ~/.nanobot/config.json，尝试自动填充凭据...${NC}"
                _merge_credentials
            else
                echo -e "${YELLOW}   请手动编辑 $NANOBOT_CONFIG 填入飞书 appId/appSecret${NC}"
                read -p "   是否继续启动? [y/N] " confirm
                if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
                    exit 1
                fi
            fi
        else
            echo -e "${GREEN}   ✓ nanobot_config.json 已配置${NC}"
        fi
        return
    fi

    # nanobot_config.json 不存在，自动生成
    echo -e "${YELLOW}   nanobot_config.json 不存在，自动生成...${NC}"

    if [ -f "$HOME_NANOBOT_CONFIG" ]; then
        echo -e "${CYAN}   从 ~/.nanobot/config.json 复制并注入 FastCode systemPrompt...${NC}"
        _generate_from_home_config
    else
        echo -e "${CYAN}   生成默认配置模板...${NC}"
        _generate_default_config
    fi

    echo -e "${GREEN}   ✓ nanobot_config.json 已生成${NC}"
}

# 从 ~/.nanobot/config.json 复制并注入 systemPrompt
_generate_from_home_config() {
    # 使用 python 来安全地操作 JSON
    python3 -c "
import json, sys

with open('$HOME_NANOBOT_CONFIG', 'r') as f:
    config = json.load(f)

# 注入 FastCode 专用 systemPrompt
if 'agents' not in config:
    config['agents'] = {}
if 'defaults' not in config['agents']:
    config['agents']['defaults'] = {}

config['agents']['defaults']['systemPrompt'] = (
    'You are FastCode Assistant, a code understanding AI that helps users '
    'analyze and query code repositories. You have access to FastCode tools '
    'that can load repositories, answer questions about code, and manage '
    'dialogue sessions. When a user sends a GitHub URL, use fastcode_load_repo '
    'to load it. When they ask about code, use fastcode_query. Always be '
    'helpful and provide clear, technical answers.'
)

# 确保 gateway 配置存在
if 'gateway' not in config:
    config['gateway'] = {'host': '0.0.0.0', 'port': 18790}

with open('$NANOBOT_CONFIG', 'w') as f:
    json.dump(config, f, indent=2, ensure_ascii=False)

print('   Done: copied credentials + injected systemPrompt')
"
}

# 将 ~/.nanobot/config.json 中的凭据合并到现有 nanobot_config.json
# 注意: API Key 和模型配置已通过 docker-compose 环境变量从 .env 统一注入，此处只合并飞书凭据
_merge_credentials() {
    python3 -c "
import json

with open('$HOME_NANOBOT_CONFIG', 'r') as f:
    home = json.load(f)
with open('$NANOBOT_CONFIG', 'r') as f:
    local = json.load(f)

changed = False

# 合并飞书凭据
home_feishu = home.get('channels', {}).get('feishu', {})
local_feishu = local.get('channels', {}).get('feishu', {})
if home_feishu.get('appId') and local_feishu.get('appId') == 'your_feishu_app_id':
    local['channels']['feishu']['appId'] = home_feishu['appId']
    local['channels']['feishu']['appSecret'] = home_feishu.get('appSecret', '')
    changed = True
    print('   ✓ 飞书 appId/appSecret 已自动填充')

# API Key 不再写入 nanobot_config.json，统一由 .env + docker-compose 环境变量注入
print('   ℹ API Key 和模型配置由 .env 统一管理 (通过 docker-compose 环境变量注入)')

if changed:
    with open('$NANOBOT_CONFIG', 'w') as f:
        json.dump(local, f, indent=2, ensure_ascii=False)
else:
    print('   (无需更新)')
"
}

# 生成默认配置模板
# 注意: API Key 和模型由 .env + docker-compose 环境变量统一管理，不写入此文件
_generate_default_config() {
    python3 -c "
import json

config = {
    'agents': {
        'defaults': {
            'workspace': '~/.nanobot/workspace',
            'maxTokens': 8192,
            'temperature': 0.7,
            'maxToolIterations': 20,
            'systemPrompt': (
                'You are FastCode Assistant, a code understanding AI that helps users '
                'analyze and query code repositories. You have access to FastCode tools '
                'that can load repositories, answer questions about code, and manage '
                'dialogue sessions. When a user sends a GitHub URL, use fastcode_load_repo '
                'to load it. When they ask about code, use fastcode_query. Always be '
                'helpful and provide clear, technical answers.'
            )
        }
    },
    'channels': {
        'feishu': {
            'enabled': True,
            'appId': 'your_feishu_app_id',
            'appSecret': 'your_feishu_app_secret',
            'encryptKey': '',
            'verificationToken': '',
            'allowFrom': []
        }
    },
    'providers': {},
    'gateway': {'host': '0.0.0.0', 'port': 18790},
    'tools': {
        'web': {'search': {'apiKey': '', 'maxResults': 5}},
        'exec': {'timeout': 60},
        'restrictToWorkspace': False
    }
}

with open('$NANOBOT_CONFIG', 'w') as f:
    json.dump(config, f, indent=2, ensure_ascii=False)

print('   已生成默认模板，请编辑 nanobot_config.json 填入飞书凭据')
print('   API Key 和模型配置请在 .env 中统一设置')
"
}

# ============ 检查 FastCode 配置文件 ============
check_fastcode_config() {
    echo -e "${BLUE}[Config] 检查 FastCode 配置...${NC}"

    if [ ! -f "$PROJECT_ROOT/.env" ]; then
        echo -e "${RED}   缺少 .env 文件${NC}"
        echo -e "   请创建 .env 并配置 OPENAI_API_KEY, MODEL, BASE_URL, NANOBOT_MODEL"
        exit 1
    fi
    echo -e "${GREEN}   ✓ .env${NC}"

    # 检查 .env 中是否有 API Key
    if grep -q '^OPENAI_API_KEY=.\+' "$PROJECT_ROOT/.env" 2>/dev/null; then
        echo -e "${GREEN}   ✓ API Key 已配置 (两个服务共用)${NC}"
    else
        echo -e "${YELLOW}   ⚠ .env 中未配置 OPENAI_API_KEY${NC}"
    fi

    if [ ! -f "$PROJECT_ROOT/config/config.yaml" ]; then
        echo -e "${RED}   缺少 config/config.yaml${NC}"
        exit 1
    fi
    echo -e "${GREEN}   ✓ config/config.yaml${NC}"
}

# ============ 从 .env 同步 API Key 和模型到 nanobot_config.json ============
# .env 是唯一的 API Key 配置源，启动前自动同步到 nanobot_config.json
sync_env_to_nanobot_config() {
    if [ ! -f "$PROJECT_ROOT/.env" ] || [ ! -f "$NANOBOT_CONFIG" ]; then
        return
    fi

    echo -e "${BLUE}[Sync] 从 .env 同步 API Key 和模型到 nanobot_config.json...${NC}"

    python3 -c "
import json

# 读取 .env 文件
env_vars = {}
with open('$PROJECT_ROOT/.env', 'r') as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            key, _, value = line.partition('=')
            env_vars[key.strip()] = value.strip()

api_key = env_vars.get('OPENAI_API_KEY', '')
nanobot_model = env_vars.get('NANOBOT_MODEL', '')

if not api_key:
    print('   (skip) .env 中未找到 OPENAI_API_KEY')
    exit(0)

# 读取 nanobot_config.json
with open('$NANOBOT_CONFIG', 'r') as f:
    config = json.load(f)

changed = False

# 同步 API Key 到 openrouter provider
providers = config.setdefault('providers', {})
openrouter = providers.setdefault('openrouter', {})
if openrouter.get('apiKey') != api_key:
    openrouter['apiKey'] = api_key
    changed = True
    print('   ✓ OpenRouter API Key 已从 .env 同步')

# 同步 Nanobot 模型
if nanobot_model:
    agents = config.setdefault('agents', {})
    defaults = agents.setdefault('defaults', {})
    if defaults.get('model') != nanobot_model:
        defaults['model'] = nanobot_model
        changed = True
        print(f'   ✓ Nanobot 模型已同步: {nanobot_model}')

if changed:
    with open('$NANOBOT_CONFIG', 'w') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
else:
    print('   ✓ 配置已是最新')
"
}

# ============ 创建必要目录 ============
ensure_dirs() {
    mkdir -p "$PROJECT_ROOT/data" "$PROJECT_ROOT/repos" "$PROJECT_ROOT/logs"
}

# ============ 构建 Docker 镜像 ============
build_images() {
    echo -e "${BLUE}[Build] 构建 Docker 镜像...${NC}"
    dc build
    echo -e "${GREEN}   ✓ 镜像构建完成${NC}"
}

# ============ 启动服务并等待就绪 ============
start_and_wait() {
    local detach_flag="${1:--d}"
    echo -e "${BLUE}[Start] 启动服务...${NC}"
    dc up $detach_flag

    if [ "$detach_flag" = "-d" ]; then
        echo -e "${YELLOW}   等待 FastCode API 就绪...${NC}"
        for i in $(seq 1 30); do
            if curl -sf http://localhost:8001/health > /dev/null 2>&1; then
                print_success_box
                return 0
            fi
            sleep 2
        done
        echo -e "${YELLOW}   服务仍在启动中，请用 $0 logs 查看日志${NC}"
    fi
}

print_success_box() {
    echo ""
    echo "╔══════════════════════════════════════════════╗"
    echo -e "║  ${GREEN}FastCode + Nanobot 已启动${NC}                   ║"
    echo "╠══════════════════════════════════════════════╣"
    echo "║  FastCode API:  http://localhost:8001        ║"
    echo "║  FastCode Docs: http://localhost:8001/docs   ║"
    echo "║  Nanobot 网关:  http://localhost:18791       ║"
    echo "║                                              ║"
    echo "║  飞书机器人已通过 WebSocket 长连接接入       ║"
    echo "╠══════════════════════════════════════════════╣"
    echo -e "║  查看日志: ${CYAN}./run_nanobot.sh logs${NC}             ║"
    echo -e "║  停止服务: ${CYAN}./run_nanobot.sh stop${NC}             ║"
    echo -e "║  查看状态: ${CYAN}./run_nanobot.sh status${NC}           ║"
    echo "╚══════════════════════════════════════════════╝"
    echo ""
}

# ============ 智能启动 (核心逻辑) ============
# 自适应检测当前状态，决定执行什么操作
smart_start() {
    local detach_flag="${1:--d}"

    print_banner
    check_docker

    # Step 1: 检测容器状态
    local fc_state nb_state
    fc_state=$(get_container_state "fastcode")
    nb_state=$(get_container_state "nanobot")

    # Case A: 两个容器都在运行
    if [ "$fc_state" = "running" ] && [ "$nb_state" = "running" ]; then
        echo -e "${GREEN}[Status] 服务已在运行中${NC}"
        echo ""
        # 健康检查
        if curl -sf http://localhost:8001/health > /dev/null 2>&1; then
            echo -e "${GREEN}   ✓ FastCode API 正常 (http://localhost:8001)${NC}"
        else
            echo -e "${YELLOW}   ⚠ FastCode API 未响应，尝试重启...${NC}"
            dc restart fastcode
        fi
        if nc -z localhost 18791 2>/dev/null; then
            echo -e "${GREEN}   ✓ Nanobot 网关正常 (http://localhost:18791)${NC}"
        else
            echo -e "${YELLOW}   ⚠ Nanobot 网关未响应，尝试重启...${NC}"
            dc restart nanobot
        fi
        echo ""
        echo -e "提示: 使用 ${CYAN}$0 restart${NC} 强制重启，${CYAN}$0 logs${NC} 查看日志"
        return 0
    fi

    # Case B: 容器存在但已停止 → 重启
    if [ "$fc_state" = "stopped" ] || [ "$nb_state" = "stopped" ]; then
        echo -e "${YELLOW}[Status] 检测到服务已停止，重新启动...${NC}"
        check_fastcode_config
        auto_configure_nanobot
        sync_env_to_nanobot_config
        ensure_dirs
        dc up -d
        echo -e "${YELLOW}   等待服务就绪...${NC}"
        for i in $(seq 1 30); do
            if curl -sf http://localhost:8001/health > /dev/null 2>&1; then
                print_success_box
                return 0
            fi
            sleep 2
        done
        echo -e "${YELLOW}   服务仍在启动中，请用 $0 logs 查看日志${NC}"
        return 0
    fi

    # Case C: 没有容器 → 检查镜像 → 可能需要构建
    echo -e "${BLUE}[Status] 未检测到运行中的服务，开始初始化...${NC}"
    echo ""

    # Step 2: 检查配置文件
    check_fastcode_config
    auto_configure_nanobot
    sync_env_to_nanobot_config
    ensure_dirs

    # Step 3: 检查镜像，按需构建
    if images_exist; then
        echo -e "${GREEN}[Build] Docker 镜像已存在，跳过构建${NC}"
    else
        echo -e "${YELLOW}[Build] 首次运行，构建 Docker 镜像...${NC}"
        build_images
    fi

    # Step 4: 启动
    start_and_wait "$detach_flag"
}

# ============ 帮助信息 ============
usage() {
    echo "用法: $0 [命令] [选项]"
    echo ""
    echo "命令:"
    echo "  (无参数)      智能启动: 自动检测状态并执行合适的操作"
    echo "  stop          停止所有服务"
    echo "  restart       重启所有服务"
    echo "  logs          查看实时日志"
    echo "  status        查看服务状态"
    echo "  clean         停止并删除容器和镜像"
    echo "  config        重新生成/检查 nanobot 配置"
    echo ""
    echo "选项:"
    echo "  --build       强制重新构建 Docker 镜像"
    echo "  --fg          前台运行 (默认后台)"
    echo "  -h, --help    显示帮助信息"
    echo ""
    echo "示例:"
    echo "  $0              # 智能启动 (首次自动构建，后续自动检测)"
    echo "  $0 --build      # 强制重新构建后启动"
    echo "  $0 --fg         # 前台运行 (可看实时日志)"
    echo "  $0 stop         # 停止服务"
    echo "  $0 restart      # 重启服务"
    echo "  $0 logs         # 查看日志"
    echo "  $0 status       # 查看状态"
    echo "  $0 config       # 重新检查/生成配置"
}

# ============ 解析命令行参数 ============
ACTION="smart"
DETACH_FLAG="-d"
FORCE_BUILD=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --build)
            FORCE_BUILD=true
            shift
            ;;
        --fg)
            DETACH_FLAG=""
            shift
            ;;
        -d|--detach)
            DETACH_FLAG="-d"
            shift
            ;;
        stop)
            ACTION="stop"
            shift
            ;;
        restart)
            ACTION="restart"
            shift
            ;;
        logs)
            ACTION="logs"
            shift
            ;;
        status)
            ACTION="status"
            shift
            ;;
        clean)
            ACTION="clean"
            shift
            ;;
        config)
            ACTION="config"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo -e "${RED}未知参数: $1${NC}"
            usage
            exit 1
            ;;
    esac
done

# ============ 主流程 ============
case $ACTION in
    smart)
        if [ "$FORCE_BUILD" = true ]; then
            print_banner
            check_docker
            check_fastcode_config
            auto_configure_nanobot
            sync_env_to_nanobot_config
            ensure_dirs
            echo -e "${YELLOW}[Build] 强制重新构建镜像...${NC}"
            build_images
            start_and_wait "$DETACH_FLAG"
        else
            smart_start "$DETACH_FLAG"
        fi
        ;;

    stop)
        check_docker
        echo -e "${BLUE}停止 FastCode + Nanobot 服务...${NC}"
        dc down
        echo -e "${GREEN}✓ 所有服务已停止${NC}"
        ;;

    restart)
        check_docker
        echo -e "${BLUE}重启 FastCode + Nanobot 服务...${NC}"
        dc down
        check_fastcode_config
        auto_configure_nanobot
        sync_env_to_nanobot_config
        ensure_dirs
        if [ "$FORCE_BUILD" = true ]; then
            build_images
        fi
        start_and_wait "-d"
        ;;

    logs)
        check_docker
        echo -e "${BLUE}服务日志 (Ctrl+C 退出):${NC}"
        echo ""
        dc logs -f
        ;;

    status)
        check_docker
        echo -e "${BLUE}服务状态:${NC}"
        echo ""
        dc ps
        echo ""
        if curl -sf http://localhost:8001/health > /dev/null 2>&1; then
            echo -e "${GREEN}✓ FastCode API 运行正常 (http://localhost:8001)${NC}"
        else
            echo -e "${YELLOW}⚠ FastCode API 未响应${NC}"
        fi
        if curl -sf http://localhost:18791 > /dev/null 2>&1 || \
           nc -z localhost 18791 2>/dev/null; then
            echo -e "${GREEN}✓ Nanobot 网关运行中 (http://localhost:18791)${NC}"
        else
            echo -e "${YELLOW}⚠ Nanobot 网关未响应${NC}"
        fi
        ;;

    config)
        echo -e "${BLUE}检查并配置 Nanobot...${NC}"
        auto_configure_nanobot
        sync_env_to_nanobot_config
        echo ""
        echo -e "配置文件: ${CYAN}$NANOBOT_CONFIG${NC}"
        echo -e "API Key 和模型配置请在 ${CYAN}.env${NC} 中统一设置"
        ;;

    clean)
        check_docker
        echo -e "${YELLOW}即将停止并删除 FastCode + Nanobot 容器和镜像${NC}"
        echo -e "${YELLOW}(数据目录 data/, repos/, logs/ 不会被删除)${NC}"
        read -p "确认? [y/N] " confirm
        if [[ "$confirm" =~ ^[Yy]$ ]]; then
            dc down --rmi local --remove-orphans -v
            echo -e "${GREEN}✓ 已清理完成${NC}"
        else
            echo "已取消"
        fi
        ;;
esac
