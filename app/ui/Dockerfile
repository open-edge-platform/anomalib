# syntax=docker/dockerfile:1.7

#############
# build_rest_api_specs
#############
FROM node:24-alpine3.22@sha256:7aaba6b13a55a1d78411a1162c1994428ed039c6bbef7b1d9859c25ada1d7cc5 AS build_rest_api_specs

WORKDIR /home/app/web_ui/

COPY --link ./package.json .
COPY --link --from=openapi_spec /openapi.json /home/app/web_ui/src/api/openapi-spec.json
RUN npm run build:api

#############
# geti_ui_packages
#############
FROM node:24-alpine3.22@sha256:7aaba6b13a55a1d78411a1162c1994428ed039c6bbef7b1d9859c25ada1d7cc5 AS geti_ui_packags

WORKDIR /home/app/web_ui/
RUN apk add git
COPY --link package.json ./
RUN npm run clone-geti-ui-packages

#############
# web_ui_deps
#############
FROM node:24-alpine3.22@sha256:7aaba6b13a55a1d78411a1162c1994428ed039c6bbef7b1d9859c25ada1d7cc5 AS base_web_ui

WORKDIR /home/app/web_ui/

COPY --link package.json ./
COPY --link package-lock.json ./
COPY --from=geti_ui_packags /home/app/web_ui/packages/ /home/app/web_ui/packages/
RUN npm ci --audit=false --ignore-scripts

COPY --link ./eslint.config.js ./
COPY --link ./.prettierrc ./
COPY --link ./tsconfig.json ./
COPY --link ./rsbuild.config.ts ./
COPY --link src/ src/

ARG PUBLIC_API_BASE_URL=""
ENV PUBLIC_API_BASE_URL ${PUBLIC_API_BASE_URL}

#############
# dev
#############
FROM base_web_ui as dev

CMD npm run dev

#############
# web_ui build
#############
FROM base_web_ui as web_ui
RUN npm run build

#######
# Nginx
#######
FROM nginx:1.27-alpine3.21@sha256:65645c7bb6a0661892a8b03b89d0743208a18dd2f3f17a54ef4b76fb8e2f2a10 AS nginx

COPY ./nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=web_ui /home/app/web_ui/dist/ /usr/share/nginx/html

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
