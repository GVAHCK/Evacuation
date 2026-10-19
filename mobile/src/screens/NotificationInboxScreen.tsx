import React, { useEffect, useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  FlatList,
  TouchableOpacity,
  RefreshControl,
  ActivityIndicator,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { useNavigation } from '@react-navigation/native';
import { LiquidGlassTokens } from '../theme/tokens';
import { GlassCard } from '../components/glass/GlassCard';
import { GlassBadge } from '../components/glass/GlassBadge';
import { GlassHeader } from '../components/glass/GlassHeader';
import { Api } from '../api/client';
import { useTheme } from '../theme/ThemeContext';

export interface NotificationItem {
  id: string;
  type: string;
  title: string;
  message: string;
  timestamp: string;
  read: boolean;
  severity: 'info' | 'success' | 'warning' | 'danger';
}

export const NotificationInboxScreen: React.FC = () => {
  const navigation = useNavigation<any>();
  const { haptic } = useTheme();
  const [notifications, setNotifications] = useState<NotificationItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [filter, setFilter] = useState<'all' | 'unread'>('all');

  const fetchNotifications = async () => {
    try {
      const data = await Api.getNotifications();
      setNotifications(data || []);
    } catch (err) {
      // Fallback
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  useEffect(() => {
    fetchNotifications();
  }, []);

  const handleMarkRead = async (id: string) => {
    haptic.selection();
    setNotifications((prev) =>
      prev.map((n) => (n.id === id ? { ...n, read: true } : n))
    );
    try {
      await Api.markNotificationRead(id);
    } catch {}
  };

  const filteredItems = notifications.filter((n) =>
    filter === 'unread' ? !n.read : true
  );

  const unreadCount = notifications.filter((n) => !n.read).length;

  return (
    <SafeAreaView style={styles.safeArea}>
      <GlassHeader
        title="Event & Alert Inbox"
        showBack
        onBackPress={() => navigation.goBack()}
        badge={
          <GlassBadge
            status={unreadCount > 0 ? 'warning' : 'healthy'}
            label={`${unreadCount} New`}
            size="sm"
          />
        }
      />

      <View style={styles.filterRow}>
        <TouchableOpacity
          style={[styles.filterBtn, filter === 'all' && styles.filterBtnActive]}
          onPress={() => setFilter('all')}
          accessibilityRole="button"
          accessibilityLabel="Show all notifications"
        >
          <Text style={[styles.filterText, filter === 'all' && styles.filterTextActive]}>
            All Events ({notifications.length})
          </Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={[styles.filterBtn, filter === 'unread' && styles.filterBtnActive]}
          onPress={() => setFilter('unread')}
          accessibilityRole="button"
          accessibilityLabel="Show unread notifications"
        >
          <Text style={[styles.filterText, filter === 'unread' && styles.filterTextActive]}>
            Unread ({unreadCount})
          </Text>
        </TouchableOpacity>
      </View>

      {loading ? (
        <View style={styles.centerContainer}>
          <ActivityIndicator size="large" color={LiquidGlassTokens.accent.primary} />
          <Text style={styles.loadingText}>Fetching event stream...</Text>
        </View>
      ) : filteredItems.length === 0 ? (
        <View style={styles.emptyContainer}>
          <Ionicons name="notifications-off-outline" size={48} color={LiquidGlassTokens.text.muted} />
          <Text style={styles.emptyTitle}>No Notifications</Text>
          <Text style={styles.emptyDesc}>
            All multi-hazard system events and operational alerts have been acknowledged.
          </Text>
        </View>
      ) : (
        <FlatList
          data={filteredItems}
          keyExtractor={(item) => item.id}
          contentContainerStyle={styles.listContent}
          refreshControl={
            <RefreshControl
              refreshing={refreshing}
              onRefresh={() => {
                setRefreshing(true);
                fetchNotifications();
              }}
              tintColor="#00D2FF"
            />
          }
          renderItem={({ item }) => {
            const timeStr = item.timestamp
              ? new Date(item.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
              : 'Recent';

            return (
              <TouchableOpacity
                activeOpacity={0.8}
                onPress={() => handleMarkRead(item.id)}
                accessibilityRole="button"
                accessibilityLabel={`${item.title}: ${item.message}`}
              >
                <GlassCard
                  style={[styles.card, !item.read && styles.cardUnread]}
                  variant={!item.read ? 'standard' : 'ultraThin'}
                >
                  <View style={styles.cardHeader}>
                    <View style={styles.titleRow}>
                      <Ionicons
                        name={
                          item.type === 'evacuation'
                            ? 'navigate-circle'
                            : item.severity === 'danger'
                            ? 'flame'
                            : 'information-circle'
                        }
                        size={20}
                        color={
                          item.severity === 'danger'
                            ? LiquidGlassTokens.accent.danger
                            : item.severity === 'success'
                            ? LiquidGlassTokens.accent.success
                            : LiquidGlassTokens.accent.primary
                        }
                      />
                      <Text style={styles.cardTitle}>{item.title}</Text>
                    </View>
                    <Text style={styles.timeText}>{timeStr}</Text>
                  </View>

                  <Text style={styles.messageText}>{item.message}</Text>

                  <View style={styles.footerRow}>
                    <GlassBadge
                      status={
                        item.severity === 'danger'
                          ? 'danger'
                          : item.severity === 'warning'
                          ? 'warning'
                          : item.severity === 'success'
                          ? 'completed'
                          : 'healthy'
                      }
                      label={item.type.toUpperCase()}
                      size="sm"
                    />
                    {!item.read && (
                      <View style={styles.unreadPill}>
                        <View style={styles.unreadDot} />
                        <Text style={styles.unreadText}>Mark Read</Text>
                      </View>
                    )}
                  </View>
                </GlassCard>
              </TouchableOpacity>
            );
          }}
        />
      )}
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  safeArea: {
    flex: 1,
    backgroundColor: LiquidGlassTokens.background.primary,
  },
  filterRow: {
    flexDirection: 'row',
    paddingHorizontal: 16,
    paddingVertical: 10,
    backgroundColor: 'rgba(11, 17, 32, 0.60)',
    borderBottomWidth: 1,
    borderBottomColor: 'rgba(255, 255, 255, 0.08)',
  },
  filterBtn: {
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: LiquidGlassTokens.radius.pill,
    marginRight: 10,
    backgroundColor: 'rgba(255, 255, 255, 0.06)',
  },
  filterBtnActive: {
    backgroundColor: 'rgba(0, 210, 255, 0.18)',
    borderColor: 'rgba(0, 210, 255, 0.40)',
    borderWidth: 1,
  },
  filterText: {
    fontSize: 13,
    color: LiquidGlassTokens.text.secondary,
    fontWeight: '600',
  },
  filterTextActive: {
    color: LiquidGlassTokens.accent.primary,
  },
  listContent: {
    padding: 16,
    paddingBottom: 40,
  },
  card: {
    padding: 16,
    marginBottom: 12,
  },
  cardUnread: {
    borderColor: 'rgba(0, 210, 255, 0.35)',
    borderTopColor: 'rgba(255, 255, 255, 0.45)',
  },
  cardHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  titleRow: {
    flexDirection: 'row',
    alignItems: 'center',
    flex: 1,
  },
  cardTitle: {
    fontSize: 15,
    fontWeight: '700',
    color: LiquidGlassTokens.text.primary,
    marginLeft: 8,
    flex: 1,
  },
  timeText: {
    fontSize: 12,
    color: LiquidGlassTokens.text.muted,
  },
  messageText: {
    fontSize: 13,
    color: LiquidGlassTokens.text.secondary,
    lineHeight: 18,
    marginBottom: 12,
  },
  footerRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  unreadPill: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  unreadDot: {
    width: 6,
    height: 6,
    borderRadius: 3,
    backgroundColor: LiquidGlassTokens.accent.primary,
    marginRight: 6,
  },
  unreadText: {
    fontSize: 11,
    color: LiquidGlassTokens.accent.primary,
    fontWeight: '600',
  },
  centerContainer: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
  },
  loadingText: {
    color: LiquidGlassTokens.text.secondary,
    marginTop: 12,
  },
  emptyContainer: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    padding: 30,
  },
  emptyTitle: {
    fontSize: 18,
    fontWeight: '700',
    color: LiquidGlassTokens.text.primary,
    marginTop: 14,
  },
  emptyDesc: {
    fontSize: 13,
    color: LiquidGlassTokens.text.secondary,
    textAlign: 'center',
    marginTop: 6,
    lineHeight: 18,
  },
});
