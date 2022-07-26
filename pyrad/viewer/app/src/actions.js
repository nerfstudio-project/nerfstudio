export const SEND_MESSAGE_REQUEST = "SEND_MESSAGE_REQUEST"
export const UPDATE_CHAT_LOG = "UPDATE_CHAT_LOG"

export function updateChatLog(update){
    return {
        type: UPDATE_CHAT_LOG,
        update
    }
}