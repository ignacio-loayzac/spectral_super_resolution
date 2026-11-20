#!/usr/bin/env bash
# Verificación de conectividad (solo lectura + 1 objeto de prueba que se borra).
# No crea pods ni genera costo. Uso: bash scripts/check_cloud.sh
set -uo pipefail
cd "$(dirname "$0")/.."
set -a; [ -f .env ] && . ./.env; set +a
ok=0

echo "== R2 (credenciales S3 usadas por los pods) =="
if [ -z "${SPECTRAL_RECONSTRUCTION_R2_ACCESS_KEY_ID:-}" ] || [[ "${SPECTRAL_RECONSTRUCTION_R2_ACCESS_KEY_ID}" == your_* ]]; then
  echo "  FALTA: SPECTRAL_RECONSTRUCTION_R2_ACCESS_KEY_ID / SECRET / ENDPOINT_URL en .env"; ok=1
else
  export RCLONE_CONFIG_R2_TYPE=s3 RCLONE_CONFIG_R2_PROVIDER=Cloudflare \
         RCLONE_CONFIG_R2_ACCESS_KEY_ID="$SPECTRAL_RECONSTRUCTION_R2_ACCESS_KEY_ID" \
         RCLONE_CONFIG_R2_SECRET_ACCESS_KEY="$SPECTRAL_RECONSTRUCTION_R2_SECRET_ACCESS_KEY" \
         RCLONE_CONFIG_R2_ENDPOINT="$SPECTRAL_RECONSTRUCTION_R2_ENDPOINT_URL" \
         RCLONE_CONFIG_R2_NO_CHECK_BUCKET=true RCLONE_LOG_LEVEL=ERROR
  D="${SPECTRAL_RECONSTRUCTION_R2_BUCKET_NAME}"; E="${SPECTRAL_RECONSTRUCTION_R2_BUCKET_NAME_EXPERIMENTS}"
  if rclone lsf "r2:$D/Models/" >/tmp/_r2models 2>&1; then echo "  lectura OK ($D/Models: $(wc -l </tmp/_r2models | tr -d ' ') archivos)"; else echo "  lectura FALLÓ: $(tail -1 /tmp/_r2models)"; ok=1; fi
  for BK in "$E" "$D"; do if echo "ping" | rclone rcat "r2:$BK/_connectivity_check/ping.txt" 2>/tmp/_r2w && rclone deletefile "r2:$BK/_connectivity_check/ping.txt" 2>>/tmp/_r2w; then echo "  escritura OK ($BK)"; else echo "  escritura FALLÓ ($BK): $(tail -1 /tmp/_r2w)"; ok=1; fi; done
fi

echo "== RunPod (permisos de la API key; no crea nada que genere costo) =="
if [ -z "${RUNPOD_API_KEY:-}" ]; then
  echo "  FALTA: RUNPOD_API_KEY en .env"; ok=1
else
  export RUNPOD_API_KEY
  probe() { # $1=etiqueta $2=necesario(1/0) resto=comando
    local label=$1 req=$2; shift 2
    local out; out=$("$@" 2>&1 | head -c 300)
    if echo "$out" | grep -qE '"status":(401|403)'; then
      echo "  [$( [ $req = 1 ] && echo FALTA || echo opcional)] $label: $(echo "$out" | python3 -c 'import sys,json;print(json.load(sys.stdin).get("error",""))' 2>/dev/null)"
      [ $req = 1 ] && ok=1
    else echo "  [OK] $label"; fi
  }
  probe "pods:read (listar pods)"              1 runpodctl pod list
  probe "pods:write (crear; GPU inválida a propósito)" 1 runpodctl pod create --name zz-permcheck --image busybox --gpu-id "NONEXISTENT GPU" --container-disk-in-gb 5
  probe "pods:write (detener/eliminar)"        1 runpodctl pod stop zzzzzzzzzzzzzz
  probe "catalog:read (GPUs, precios, stock)"  1 runpodctl gpu list
  probe "billing:read (historial de gasto)"     1 runpodctl billing pods
  probe "cuenta vía GraphQL (saldo y límite de gasto)" 0 runpodctl user
  probe "registry:read"                        0 runpodctl registry list
  probe "network-volume:read"                  0 runpodctl network-volume list
  # templates:write crea una plantilla real (gratis): se borra inmediatamente.
  tout=$(runpodctl template create --name zz-permcheck --image busybox --container-disk-in-gb 1 2>&1)
  tid=$(echo "$tout" | python3 -c 'import sys,json;print(json.load(sys.stdin).get("id",""))' 2>/dev/null)
  if [ -n "$tid" ]; then runpodctl template delete "$tid" >/dev/null 2>&1; echo "  [OK] templates:write (plantilla de prueba creada y borrada)"
  else echo "  [opcional] templates:write: $(echo "$tout" | head -c 150)"; fi
fi

echo "== Imagen GHCR =="
code=$(curl -s -o /dev/null -w '%{http_code}' -H "Authorization: Bearer $(curl -s 'https://ghcr.io/token?scope=repository:ignacio-loayzac/spectral_super_resolution:pull' | python3 -c 'import sys,json;print(json.load(sys.stdin).get("token",""))')" \
       https://ghcr.io/v2/ignacio-loayzac/spectral_super_resolution/manifests/${IMAGE_TAG:-1.0.1} -H 'Accept: application/vnd.oci.image.index.v1+json, application/vnd.docker.distribution.manifest.v2+json')
[ "$code" = 200 ] && echo "  imagen :${IMAGE_TAG:-1.0.1} pública y accesible (ghcr.io/ignacio-loayzac/spectral_super_resolution)" || echo "  imagen :${IMAGE_TAG:-1.0.1} no accesible anónimamente (HTTP $code): privada o inexistente"

exit $ok
