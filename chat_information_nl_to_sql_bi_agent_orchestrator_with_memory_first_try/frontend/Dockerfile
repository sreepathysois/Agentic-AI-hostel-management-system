# frontend/Dockerfile
FROM node:20-alpine

WORKDIR /app
COPY package.json .
RUN npm install
COPY . .
RUN npm run build

# serve with a simple static server
RUN npm install -g serve
EXPOSE 3000
CMD ["serve", "-s", "dist", "-l", "3000"]

